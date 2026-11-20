import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="card p-10 text-center">
      <h1>Page not found</h1>
      <p className="mt-2 text-fg-2"><Link to="/">Back to projects</Link></p>
    </div>
  );
}
