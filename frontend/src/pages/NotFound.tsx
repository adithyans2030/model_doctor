import { Compass } from "lucide-react";
import { Link } from "react-router-dom";
import { EmptyState, buttonClasses } from "@/components/ui";

export default function NotFound() {
  return (
    <div className="mx-auto mt-10 max-w-md rounded-lg border border-line bg-surface">
      <EmptyState icon={Compass} title="Page not found" description="The link may be out of date, or the item was deleted."
        action={<Link to="/app" className={buttonClasses()}>Back to projects</Link>} />
    </div>
  );
}
