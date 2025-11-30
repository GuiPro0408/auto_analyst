from tools.parser import clean_text, parse_html


def test_parse_html_strips_scripts():
    html = "<html><head><title>Hi</title><script>ignore()</script></head><body><p>Hello</p></body></html>"
    title, text = parse_html(html)
    assert title == "Hi"
    assert "Hello" in text
    assert "ignore" not in text


def test_clean_text_condenses_whitespace():
    assert clean_text("a   b\\n c") == "a b c"
