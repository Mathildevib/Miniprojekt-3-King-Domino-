import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from collections import deque

# Variables for Ground Truth CSV, GT for scores as well as the folder for gameboards all in CAPS for constants, add own paths here
CSV_PATH = r"C:\Users\alexz\Desktop\AI_programmering\cropped_tiles\tile_gt_with_crowns.csv"
BOARD_FOLDER = r"C:\Users\alexz\Desktop\King_Domino\Data"
SCORE_CSV = r"C:\Users\alexz\Desktop\AI_programmering\cropped_tiles\king_domino_scores.csv"

# Creating Class for processing the game boards and extract features
class BoardProcessor:
    def __init__(self, board_folder):
        self.board_folder = board_folder

# Cropping boards into individual tiles in 100x100 pixels.
    def crop_board(self, img):
        tiles = []
        for row in range(5):
            for col in range(5):
                tile = img[row * 100:(row + 1) * 100, col * 100:(col + 1) * 100]
                tiles.append((tile, col * 100, row * 100))
        return tiles

#Feature extraction using Normalization, Gaussian blur, HSV variance, HSV means, as well as LBP for both masked and unmasked tiles
    def extract_features(self, tile):
        tile = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX)
        tile = cv2.GaussianBlur(tile, (5, 5), 0)
        mask = np.ones((100, 100), dtype="uint8") * 255
        mask[25:75, 25:75] = 0

        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        hsv_vars = [np.var(hsv[mask > 0, i]) for i in range(3)]
        hsv_means = [np.mean(hsv[mask > 0, i]) for i in range(3)]

        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        lbp_r1 = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_r2 = local_binary_pattern(gray, P=16, R=2, method="uniform")
        lbp_r3 = local_binary_pattern(gray, P=24, R=3, method="uniform")

        def lbp_hist(lbp):
            return np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)[0], \
                   np.histogram(lbp[mask > 0], bins=26, range=(0, 26), density=True)[0]

        h1f, h1e = lbp_hist(lbp_r1)
        h2f, h2e = lbp_hist(lbp_r2)
        h3f, h3e = lbp_hist(lbp_r3)

        return np.concatenate([hsv_vars, hsv_means, h1f, h1e, h2f, h2e, h3f, h3e])

# Loads board images and cropped tiles, the extracted features and matching GT.
# Returnsthe feature vectors, tile metadata and handles errors for missing or invalid data.   
    def load_split_data(self, ids, df):
        X, y, info = [], [], []
        for image_id in ids:
            board_path = os.path.join(self.board_folder, f"{image_id}.jpg")
            if not os.path.exists(board_path):
                continue
            img = cv2.imread(board_path)
            if img is None:
                continue
            tiles = self.crop_board(img)
            board_df = df[df["image_id"] == image_id]
            for tile, x, y_pos in tiles:
                gt_row = board_df[(board_df["x"] == x) & (board_df["y"] == y_pos)]
                if gt_row.empty:
                    continue
                label = gt_row["label"].values[0]
                if label in ["Home", "Unknown"]:
                    label += " 0"
                X.append(self.extract_features(tile))
                y.append(label)
                info.append((image_id, x, y_pos))
        return np.array(X), np.array(y), info

#Creating class to train the model with SVM
class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(kernel='rbf', gamma='auto', class_weight=None, C=10,
                         probability=True, random_state=42)

    def train(self, X_train, y_train, X_val, y_val):
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])
        X_trainval_scaled = self.scaler.fit_transform(X_trainval)
        self.model.fit(X_trainval_scaled, y_trainval)
        return X_trainval_scaled, y_trainval

    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        print("Classification Report on Test Set:")
        print(classification_report(y_test, y_pred))
        return X_test_scaled

    def predict(self, features):
        scaled_features = self.scaler.transform([features])
        return self.model.predict(scaled_features)[0]

#Create Class to calculate scores using BFS
class ScoreCalculator:
    @staticmethod
    def parse_label(label):
        parts = label.split()
        return parts[0], int(parts[1]) if len(parts) > 1 else 0

    def calculate_score(self, board_df):
        grid = [[None for _ in range(5)] for _ in range(5)]
        visited = [[False for _ in range(5)] for _ in range(5)]
        score = 0
        bonus_home = False

        for _, row in board_df.iterrows():
            x, y, label = row['x'], row['y'], row['label']
            col, row_idx = x // 100, y // 100
            terrain, crowns = self.parse_label(label)
            grid[row_idx][col] = (terrain, crowns)
            if x == 200 and y == 200 and terrain == "Home":
                bonus_home = True

        def bfs(i, j, terrain_type):
            queue = deque([(i, j)])
            visited[i][j] = True
            count, crown_sum = 0, 0
            while queue:
                ci, cj = queue.popleft()
                t, c = grid[ci][cj]
                count += 1
                crown_sum += c
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < 5 and 0 <= nj < 5 and not visited[ni][nj]:
                        if grid[ni][nj] and grid[ni][nj][0] == terrain_type:
                            visited[ni][nj] = True
                            queue.append((ni, nj))
            return count * crown_sum

        for i in range(5):
            for j in range(5):
                if not visited[i][j] and grid[i][j]:
                    terrain, _ = grid[i][j]
                    if terrain not in ["Home", "Unknown"]:
                        score += bfs(i, j, terrain)

        if all(cell and cell[0] != "Unknown" for row in grid for cell in row):
            score += 5
        if bonus_home:
            score += 10
        return score

#Predicted scores vs actual scores 
    def compute_scores(self, test_ids, processor, trainer):
        actual_scores = pd.read_csv(SCORE_CSV).set_index("image_id")
        predicted_scores = []

        for image_id in test_ids:
            board_path = os.path.join(processor.board_folder, f"{image_id}.jpg")
            if not os.path.exists(board_path):
                continue
            img = cv2.imread(board_path)
            tiles = processor.crop_board(img)

            predicted_labels = []
            tile_coordinates = []

            for tile, x, y_pos in tiles:
                features = processor.extract_features(tile)
                predicted_label = trainer.predict(features)
                predicted_labels.append(predicted_label)
                tile_coordinates.append((x,y_pos))

            pred_df = pd.DataFrame({
                "image_id": image_id,
                "x": [x for x, y in tile_coordinates],
                "y": [y for x, y in tile_coordinates],
                "label": predicted_labels
            })

            predicted_score = self.calculate_score(pred_df)
            predicted_scores.append((image_id, predicted_score))

        predicted_df = pd.DataFrame(predicted_scores, columns=["image_id", "predicted_score"]).set_index("image_id")
        comparison = predicted_df.join(actual_scores, how='inner')
        comparison["score_diff"] = (comparison["predicted_score"] - comparison["score"]).abs()

        print("\nPredicted vs Actual Scores:")
        print(comparison)
        print(f"\nAverage Absolute Score Difference: {comparison['score_diff'].mean():.2f}")

        return comparison

#Main function, handling the overall execution of the code 
def main():
    # Load CSV
    df = pd.read_csv(CSV_PATH)

    # Split Data
    image_ids = df["image_id"].unique()
    train_val_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.125, random_state=42)

    # Initialize classes
    processor = BoardProcessor(BOARD_FOLDER)
    trainer = ModelTrainer()
    calculator = ScoreCalculator()

    # Load features
    X_train, y_train, _ = processor.load_split_data(train_ids, df)
    X_val, y_val, _ = processor.load_split_data(val_ids, df)
    X_test, y_test, _ = processor.load_split_data(test_ids, df)

    # Train and evaluate model
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.evaluate(X_test, y_test)

    # Compute scores
    comparison = calculator.compute_scores(test_ids, processor, trainer)

    return comparison

if __name__ == "__main__":
    comparison_df = main()