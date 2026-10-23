"""Check local Markdown/image links in the owned platform documentation."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]


class HtmlLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('src', 'href') and value:
                self.links.append(value)


def main():
    parser = MarkdownIt()
    failures, count = [], 0
    files = [ROOT/'README.md', *sorted((ROOT/'docs/platform').rglob('*.md'))]
    for path in files:
        links = []
        for token in parser.parse(path.read_text(encoding='utf8')):
            for child in [token, *(token.children or [])]:
                for key in ('href', 'src'):
                    if child.attrGet(key):
                        links.append(child.attrGet(key))
                if child.type in ('html_block', 'html_inline'):
                    html = HtmlLinks()
                    html.feed(child.content)
                    links.extend(html.links)
        for link in links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            count += 1
            target = (path.parent/unquote(url.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                failures.append(f'{path.relative_to(ROOT)}: missing or external local path {link}')
    for failure in failures:
        print(failure)
    print(f'{len(files)} documents, {count} local links, {len(failures)} failures')
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
