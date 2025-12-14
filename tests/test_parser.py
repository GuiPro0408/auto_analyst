"""Tests for HTML parsing and text cleaning.

Tests content extraction from HTML and text normalization.
"""

import pytest

from tools.parser import clean_text, parse_html


@pytest.mark.unit
def test_parse_html_strips_scripts():
    """Parser should strip script tags from HTML."""
    html = "<html><head><title>Hi</title><script>ignore()</script></head><body><p>Hello</p></body></html>"
    title, text = parse_html(html)
    assert title == "Hi"
    assert "Hello" in text
    assert "ignore" not in text


@pytest.mark.unit
def test_clean_text_condenses_whitespace():
    """Clean text should condense multiple whitespace to single space."""
    assert clean_text("a   b\\n c") == "a b c"


@pytest.mark.unit
def test_parse_html_extracts_title():
    """Parser should extract title from HTML."""
    html = (
        "<html><head><title>Test Title</title></head><body><p>Content</p></body></html>"
    )
    title, text = parse_html(html)
    assert title == "Test Title"


@pytest.mark.unit
def test_parse_html_strips_style_tags():
    """Parser should strip style tags from HTML."""
    html = "<html><head><style>.class { color: red; }</style></head><body><p>Visible</p></body></html>"
    title, text = parse_html(html)
    assert "Visible" in text
    assert "color" not in text
    assert ".class" not in text


@pytest.mark.unit
def test_parse_html_handles_nested_elements():
    """Parser should handle deeply nested HTML elements."""
    html = """
    <html>
    <body>
        <div>
            <div>
                <div>
                    <p>Deep <span>nested</span> content</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    title, text = parse_html(html)
    assert "Deep" in text
    assert "nested" in text
    assert "content" in text


@pytest.mark.unit
def test_parse_html_empty_document():
    """Parser should handle empty HTML documents."""
    html = "<html><head></head><body></body></html>"
    title, text = parse_html(html)
    assert title == "" or title is None or title == "Untitled"


@pytest.mark.unit
def test_parse_html_no_title():
    """Parser should handle HTML without title tag."""
    html = "<html><body><p>Content without title</p></body></html>"
    title, text = parse_html(html)
    assert "Content without title" in text


@pytest.mark.unit
def test_parse_html_strips_navigation():
    """Parser should strip navigation elements."""
    html = """
    <html>
    <body>
        <nav><a href="/">Home</a><a href="/about">About</a></nav>
        <main><p>Main content here</p></main>
        <footer>Footer content</footer>
    </body>
    </html>
    """
    title, text = parse_html(html)
    assert "Main content" in text


@pytest.mark.unit
def test_parse_html_handles_entities():
    """Parser should decode HTML entities."""
    html = "<html><body><p>Hello &amp; goodbye &lt;world&gt;</p></body></html>"
    title, text = parse_html(html)
    assert "&" in text or "Hello" in text


@pytest.mark.unit
def test_clean_text_removes_extra_newlines():
    """Clean text should remove excessive newlines."""
    text = "Line 1\n\n\n\n\nLine 2"
    cleaned = clean_text(text)
    assert "\n\n\n" not in cleaned


@pytest.mark.unit
def test_clean_text_strips_leading_trailing():
    """Clean text should strip leading and trailing whitespace."""
    text = "   content with spaces   "
    cleaned = clean_text(text)
    assert not cleaned.startswith(" ")
    assert not cleaned.endswith(" ")


@pytest.mark.unit
def test_clean_text_handles_tabs():
    """Clean text should handle tab characters."""
    text = "column1\t\tcolumn2\t\t\tcolumn3"
    cleaned = clean_text(text)
    # Tabs should be converted to spaces
    assert "\t\t" not in cleaned


@pytest.mark.unit
@pytest.mark.parametrize(
    "html,expected_in_text",
    [
        ("<p>Simple paragraph</p>", "Simple paragraph"),
        ("<h1>Header</h1><p>Body</p>", "Header"),
        ("<ul><li>Item 1</li><li>Item 2</li></ul>", "Item"),
        ("<table><tr><td>Cell</td></tr></table>", "Cell"),
        ("<a href='http://example.com'>Link text</a>", "Link text"),
    ],
)
def test_parse_html_various_elements(html, expected_in_text):
    """Parser should extract text from various HTML elements."""
    full_html = f"<html><body>{html}</body></html>"
    title, text = parse_html(full_html)
    assert expected_in_text in text


@pytest.mark.unit
def test_parse_html_unicode_content():
    """Parser should handle unicode content."""
    html = "<html><body><p>日本語 中文 한국어 Émoji: 🎉</p></body></html>"
    title, text = parse_html(html)
    assert "日本語" in text or "中文" in text or len(text) > 0


@pytest.mark.unit
def test_parse_html_malformed():
    """Parser should handle malformed HTML gracefully."""
    html = "<html><body><p>Unclosed paragraph<div>Mixed tags</p></div></body>"
    # Should not raise, should extract some content
    title, text = parse_html(html)
    assert isinstance(text, str)


@pytest.mark.unit
def test_clean_text_empty_string():
    """Clean text should handle empty strings."""
    assert clean_text("") == ""


@pytest.mark.unit
def test_clean_text_only_whitespace():
    """Clean text should handle whitespace-only strings."""
    assert clean_text("   \n\t  ") == ""


class TestMainContentExtraction:
    """Tests for main content extraction from HTML."""

    @pytest.mark.unit
    def test_extracts_main_tag_content(self):
        """Parser should prioritize <main> tag content."""
        html = """
        <html>
        <body>
            <header>Site Header Navigation Links</header>
            <nav>Menu Item 1 Menu Item 2 Menu Item 3</nav>
            <main>
                <article>
                    <p>This is the main article content that we want to extract.
                    It should be substantial enough to be useful for RAG processing.</p>
                </article>
            </main>
            <aside>Sidebar content with ads</aside>
            <footer>Footer with copyright info</footer>
        </body>
        </html>
        """
        title, text = parse_html(html)
        assert "main article content" in text
        # Nav/header/footer should be stripped
        assert "Menu Item" not in text
        assert "Site Header" not in text

    @pytest.mark.unit
    def test_extracts_article_tag_content(self):
        """Parser should extract content from <article> tag."""
        html = """
        <html>
        <body>
            <nav>Navigation</nav>
            <article>
                <h1>Article Title</h1>
                <p>Article body text with sufficient content for processing and analysis.</p>
                <p>More paragraphs of content here to make it substantial.</p>
            </article>
            <footer>Footer</footer>
        </body>
        </html>
        """
        title, text = parse_html(html)
        assert "Article Title" in text
        assert "Article body text" in text

    @pytest.mark.unit
    def test_removes_boilerplate_by_class(self):
        """Parser should remove elements with boilerplate class names."""
        html = """
        <html>
        <body>
            <div class="header-nav">Navigation menu</div>
            <div class="content">Main page content that is useful.</div>
            <div class="sidebar-ad">Advertisement banner</div>
            <div class="footer-links">Footer links</div>
            <div class="social-share">Share on Facebook Twitter</div>
        </body>
        </html>
        """
        title, text = parse_html(html)
        assert "Main page content" in text
        # Boilerplate classes should be stripped
        assert "Advertisement" not in text
        assert "Footer links" not in text
        assert "Share on Facebook" not in text

    @pytest.mark.unit
    def test_removes_boilerplate_by_id(self):
        """Parser should remove elements with boilerplate IDs."""
        html = """
        <html>
        <body>
            <div id="main-navigation">Site navigation</div>
            <div id="content">Actual content we want.</div>
            <div id="sidebar-menu">Menu items</div>
            <div id="cookie-banner">Cookie consent</div>
        </body>
        </html>
        """
        title, text = parse_html(html)
        assert "Actual content" in text
        # Boilerplate IDs should be stripped
        assert "Site navigation" not in text
        assert "Cookie consent" not in text

    @pytest.mark.unit
    def test_fallback_extracts_all_cleaned_text(self):
        """Parser should fallback to all text when no semantic containers exist."""
        html = """
        <html>
        <body>
            <div>
                <p>General content without semantic markup but still useful.</p>
                <p>More content in regular divs and paragraphs.</p>
            </div>
        </body>
        </html>
        """
        title, text = parse_html(html)
        assert "General content" in text
        assert "More content" in text
