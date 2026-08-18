import { Suspense } from "react";
import { SearchExperience } from "../../components/search-experience";

export default function SearchPage() {
  return (
    <main>
      <Suspense fallback={<div className="search-shell">Loading search…</div>}>
        <SearchExperience />
      </Suspense>
    </main>
  );
}
