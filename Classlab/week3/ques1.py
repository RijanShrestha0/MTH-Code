import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = "Movies Data.csv"
RANDOM_STATE = 42
SAMPLE_SIZE = 18  # Random sample between 15 and 20


def strength_label(corr: float) -> str:
	"""Return a qualitative strength label for a Pearson correlation."""
	abs_corr = abs(corr)
	if abs_corr >= 0.8:
		return "very strong"
	if abs_corr >= 0.6:
		return "strong"
	if abs_corr >= 0.4:
		return "moderate"
	if abs_corr >= 0.2:
		return "weak"
	return "very weak"


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
	"""Compute pairwise cosine similarity matrix using NumPy."""
	norms = np.linalg.norm(vectors, axis=1, keepdims=True)
	norms[norms == 0] = 1.0
	normalized = vectors / norms
	return normalized @ normalized.T


def analyze_correlations(work: pd.DataFrame) -> dict:
	"""Calculate and print correlation of gross with budget, votes, score."""
	print("=== Correlation Analysis (Dependent variable: gross) ===")
	print(f"Rows used: {len(work)}")

	target = "gross"
	predictors = ["budget", "votes", "score"]
	correlations = {col: work[col].corr(work[target]) for col in predictors}

	for col, corr in correlations.items():
		direction = "positive" if corr >= 0 else "negative"
		print(
			f"Correlation(gross, {col}) = {corr:.4f} -> "
			f"{strength_label(corr)} {direction} relationship"
		)

	print("\nInterpretation:")
	strongest = max(correlations, key=lambda c: abs(correlations[c]))
	print(f"- '{strongest}' has the strongest linear relationship with gross.")
	print("- Positive correlation means larger values tend to come with higher gross.")
	print("- Correlation does not prove causation; it only measures linear association.")

	return correlations


def sample_and_compare_movies(work: pd.DataFrame) -> pd.DataFrame:
	"""Sample movies, compute cosine similarity, and print key pair results."""
	print("\n=== Cosine Similarity Analysis on Movie Vectors ===")

	sample = work.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
	vectors = sample[["budget", "votes", "gross"]].to_numpy(dtype=float)
	similarity = cosine_similarity_matrix(vectors)

	pairs = []
	for i, j in itertools.combinations(range(len(sample)), 2):
		pairs.append((i, j, similarity[i, j]))

	pairs_sorted = sorted(pairs, key=lambda item: item[2])
	most_dissimilar = pairs_sorted[:3]
	most_similar = pairs_sorted[-3:][::-1]

	print(f"Random sample size: {len(sample)} movies")
	print("Sampled movie names:")
	print(", ".join(sample["name"].tolist()))

	print("\nTop 3 most similar movie pairs (cosine similarity near 1):")
	for i, j, sim in most_similar:
		print(f"- {sample.loc[i, 'name']}  <->  {sample.loc[j, 'name']}: {sim:.4f}")

	print("\nTop 3 most dissimilar movie pairs (lowest cosine similarity):")
	for i, j, sim in most_dissimilar:
		print(f"- {sample.loc[i, 'name']}  <->  {sample.loc[j, 'name']}: {sim:.4f}")

	print("\nSimilarity interpretation:")
	print("- Higher cosine similarity means movies share similar budget-votes-gross proportions.")
	print("- Lower cosine similarity means their budget-votes-gross profile differs more.")

	return sample


def plot_movie_vectors(sample: pd.DataFrame) -> None:
	"""Visualize sampled movie vectors in 3D using matplotlib."""
	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection="3d")

	genres = sample["genre"].astype("category")
	genre_codes = genres.cat.codes
	scatter = ax.scatter(
		sample["budget"],
		sample["votes"],
		sample["gross"],
		c=genre_codes,
		s=sample["score"] * 12,
		cmap="tab20",
		alpha=0.8,
	)

	ax.set_title("3D Movie Vectors: [budget, votes, gross]")
	ax.set_xlabel("budget")
	ax.set_ylabel("votes")
	ax.set_zlabel("gross")

	handles, _ = scatter.legend_elements()
	ax.legend(handles, genres.cat.categories.tolist(), title="genre", loc="best")

	plt.tight_layout()
	plt.show()


def main() -> None:
	df = pd.read_csv(DATA_FILE)

	cols = ["name", "genre", "budget", "votes", "score", "gross"]
	work = df[cols].copy()

	for col in ["budget", "votes", "score", "gross"]:
		work[col] = pd.to_numeric(work[col], errors="coerce")

	work = work.dropna(subset=["name", "budget", "votes", "score", "gross"])

	analyze_correlations(work)
	sample = sample_and_compare_movies(work)
	print("\nOpening 3D vector plot with matplotlib...")
	plot_movie_vectors(sample)


if __name__ == "__main__":
	main()
