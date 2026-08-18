import { CorpusOverview } from "../../components/corpus-overview";

export default function CorpusPage() {
  return <main><section className="search-shell corpus-page"><div className="search-heading"><span className="eyebrow">Corpus intelligence</span><h1>Know the evidence base before trusting the models.</h1><p>Coverage, missing metadata, section quality, duplicate signals, and embedding progress are first-class product metrics in AidLens.</p></div><CorpusOverview /></section></main>;
}
