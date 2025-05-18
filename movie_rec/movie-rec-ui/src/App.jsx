import { useState } from "react";
import axios from "./api/client";

function App() {
  const [title, setTitle] = useState("");
  const [topN, setTopN] = useState(5);
  const [sortBy, setSortBy] = useState("popularity");
  const [genreFilter, setGenreFilter] = useState("");
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");

  const fetchRecommendations = async () => {
    if (!title.trim()) return;

    try {
      const res = await axios.get("/recommend", {
        params: {
          title,
          top_n: topN,
          sort_by: sortBy,
          genre_filter: genreFilter || undefined,
        },
      });

      if (res.data?.recommendations) {
        setResults(res.data.recommendations);
        setError("");
      } else {
        setResults([]);
        setError("No recommendations found.");
      }
    } catch (err) {
      console.error("API error:", err);
      setError("Something went wrong. Please try again.");
      setResults([]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800">
      <div className="max-w-3xl mx-auto p-6">
        <h1 className="text-3xl font-bold text-center text-blue-700 mb-6">
          Movie Recommender 🎬
        </h1>

        <div className="bg-white rounded shadow p-4 space-y-4">
          <input
            type="text"
            className="w-full p-2 border rounded"
            placeholder="Enter movie title..."
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />

          <div className="flex gap-4">
            <input
              type="number"
              min="1"
              max="10"
              className="w-20 p-2 border rounded"
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
            />

            <select
              className="flex-1 p-2 border rounded"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="popularity">Sort by Popularity</option>
              <option value="vote_average">Sort by Rating</option>
            </select>
          </div>

          <input
            type="text"
            className="w-full p-2 border rounded"
            placeholder="Genre (optional)"
            value={genreFilter}
            onChange={(e) => setGenreFilter(e.target.value)}
          />

          <button
            className="w-full bg-blue-600 hover:bg-blue-700 text-white p-2 rounded font-semibold"
            onClick={fetchRecommendations}
          >
            Get Recommendations
          </button>
        </div>

        {error && <p className="text-red-600 mt-4">{error}</p>}

        <div className="mt-6 space-y-4">
          {results.map((movie, idx) => (
            <div key={idx} className="flex gap-4 bg-white shadow rounded p-4">
              {movie.poster && (
                <img
                  src={movie.poster}
                  alt={movie.title}
                  className="w-24 h-36 object-cover rounded"
                />
              )}
              <div>
                <h2 className="text-xl font-bold">{movie.title}</h2>
                <p className="text-sm text-gray-600 mt-1">{movie.overview}</p>
                <p className="text-xs mt-1">
                  ⭐ {movie.vote_average} | 🔥 {movie.popularity}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
