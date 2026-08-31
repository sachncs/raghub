/**
 * Web search provider.
 *
 * `DuckDuckGoSearch` is the default — it scrapes the HTML search
 * page (no API key required). The interface is intentionally
 * narrow so it can be swapped for SerpAPI / Brave / Tavily in a
 * follow-up commit.
 */

export interface WebHit {
  readonly title: string;
  readonly url: string;
  readonly snippet: string;
}

export interface WebSearchResult {
  readonly query: string;
  readonly hits: readonly WebHit[];
  readonly took: number;
}

export interface WebSearchOptions {
  readonly query: string;
  readonly maxResults?: number;
  readonly signal?: AbortSignal;
  readonly userAgent?: string;
}

export interface WebSearch {
  search(opts: WebSearchOptions): Promise<WebSearchResult>;
}

const DEFAULT_UA = 'revex/1.0 (+https://github.com/sachncs/revex)';
const DEFAULT_MAX = 8;

const decode = (s: string): string =>
  s
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/&#x27;/g, "'")
    .replace(/&#x2F;/g, '/');

const stripTags = (html: string): string =>
  decode(html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim());

export class DuckDuckGoSearch implements WebSearch {
  public async search(opts: WebSearchOptions): Promise<WebSearchResult> {
    const start = Date.now();
    const max = opts.maxResults ?? DEFAULT_MAX;
    const ua = opts.userAgent ?? DEFAULT_UA;
    const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(opts.query)}`;
    let res: Response;
    try {
      res = await fetch(url, {
        method: 'GET',
        headers: { 'user-agent': ua, accept: 'text/html' },
        ...(opts.signal ? { signal: opts.signal } : {}),
      });
    } catch (e) {
      return { query: opts.query, hits: [], took: Date.now() - start };
    }
    if (!res.ok) {
      return { query: opts.query, hits: [], took: Date.now() - start };
    }
    const html = await res.text();
    const hits = parseDuckDuckGo(html, max);
    return { query: opts.query, hits, took: Date.now() - start };
  }
}

const parseDuckDuckGo = (html: string, max: number): WebHit[] => {
  const hits: WebHit[] = [];
  const resultRe = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  const snippetRe = /<a class="result__snippet"[^>]*>([\s\S]*?)<\/a>/gi;
  let m: RegExpExecArray | null;
  const titles: { url: string; title: string }[] = [];
  while ((m = resultRe.exec(html)) !== null) {
    const url = decode(m[1] ?? '');
    const title = stripTags(m[2] ?? '');
    if (url) titles.push({ url, title });
  }
  const snippets: string[] = [];
  while ((m = snippetRe.exec(html)) !== null) {
    snippets.push(stripTags(m[1] ?? ''));
  }
  for (let i = 0; i < Math.min(titles.length, max); i++) {
    const t = titles[i];
    if (!t) continue;
    hits.push({ url: t.url, title: t.title, snippet: snippets[i] ?? '' });
  }
  return hits;
};

export const createDuckDuckGoSearch = (): WebSearch => new DuckDuckGoSearch();