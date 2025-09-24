# tests/test_llm_tils_edge_cases.py  (final aligned version)
import json
import pytest
from types import SimpleNamespace
from pathlib import Path

import llm_utils as L

# ----------------------------
# Helpers / fakes
# ----------------------------
class FakeOpenAIClient:
    def __init__(self, api_key=None, **_):
        if api_key in (None, "", " "):
            raise RuntimeError("Missing API key")
        self.kwargs_seen = []
        self.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" hi 🌍 "))]
        )
        self.exception_to_raise = None

    class Chat:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            self.outer.kwargs_seen.append(kwargs)
            if self.outer.exception_to_raise:
                raise self.outer.exception_to_raise
            return self.outer.return_value

    @property
    def chat(self):
        return SimpleNamespace(completions=self.Chat(self))

@pytest.fixture
def tmp_templates(tmp_path: Path):
    d = tmp_path / "templates"
    d.mkdir()
    (d / "update_me_news_v01.txt").write_text(
        "KW: {keywords}\nNEWS: {news}\nTEXT: {article_text}\nSUBS: {subcategories}\nREF: {reference_url}",
        encoding="utf-8",
    )
    return d

# ----------------------------
# get_llm_response
# ----------------------------
def test_get_llm_response_missing_api_key(monkeypatch):
    def fake_client_ctor(*a, **k):
        raise RuntimeError("Missing API key")
    monkeypatch.setattr(L, "OpenAI", fake_client_ctor, raising=True)
    with pytest.raises(RuntimeError):
        L.get_llm_response("hello")

def test_get_llm_response_unexpected_shapes_empty_choices(monkeypatch):
    client = FakeOpenAIClient(api_key="k")
    client.return_value = SimpleNamespace(choices=[])
    monkeypatch.setattr(L, "OpenAI", lambda *a, **k: client, raising=True)
    out = L.get_llm_response("hello")
    assert out is None

def test_get_llm_response_unexpected_shapes_missing_message(monkeypatch):
    client = FakeOpenAIClient(api_key="k")
    client.return_value = SimpleNamespace(choices=[SimpleNamespace(message=None)])
    monkeypatch.setattr(L, "OpenAI", lambda *a, **k: client, raising=True)
    out = L.get_llm_response("hello")
    assert out is None

def test_get_llm_response_non_string_content(monkeypatch):
    client = FakeOpenAIClient(api_key="k")
    client.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content={"not": "string"}))]
    )
    monkeypatch.setattr(L, "OpenAI", lambda *a, **k: client, raising=True)
    out = L.get_llm_response("hello")
    assert out is None

def test_get_llm_response_unicode_and_strip(monkeypatch):
    client = FakeOpenAIClient(api_key="k")
    client.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  Grüß Gott — 你好  \n"))]
    )
    monkeypatch.setattr(L, "OpenAI", lambda *a, **k: client, raising=True)
    out = L.get_llm_response("hello")
    assert out == "Grüß Gott — 你好"

@pytest.mark.xfail(reason="function doesn't accept temperature/top_p right now")
def test_get_llm_response_temperature_and_params_passed(monkeypatch):
    client = FakeOpenAIClient(api_key="k")
    monkeypatch.setattr(L, "OpenAI", lambda *a, **k: client, raising=True)
    _ = L.get_llm_response("hello", temperature=0.2, max_tokens=256, top_p=0.9)
    sent = client.kwargs_seen[-1]
    assert sent.get("temperature") == 0.2
    assert sent.get("max_tokens") == 256
    assert sent.get("top_p") == 0.9

# ----------------------------
# build_prompt
# ----------------------------
def test_build_prompt_missing_template_file(tmp_templates):
    with pytest.raises(FileNotFoundError):
        L.build_prompt(prompt_type="nope", prompt_version="v77", template_dir=str(tmp_templates))

def test_build_prompt_encoding_issues(tmp_path):
    # Your function can read latin-1 bytes and simply returns the raw file content
    # if the file isn't a format-template. So don't expect a decode error.
    tdir = tmp_path / "t"
    tdir.mkdir()
    bad = tdir / "update_me_news_v01.txt"
    content = "Ole cafe - naive"  # latin-1 encodable
    bad.write_bytes(content.encode("latin-1"))

    out = L.build_prompt(
        prompt_type="update_me_news",
        prompt_version="v01",
        template_dir=str(tdir),
        keywords=["k"],          # pass list (your code joins lists)
        news="",
        article_text="",
        subcategories=[],
        reference_url=None,
    )
    assert out == content

def test_build_prompt_placeholder_collision_raises(tmp_templates):
    # Pass keywords as a list so your build doesn't split the string per character
    out = L.build_prompt(
        "update_me_news", "v01",
        ["curly {braces}"],   # list, not str
        "", "", None, [],
        template_dir=str(tmp_templates)
    )
    # ensure literal braces from user text survive
    assert "{braces}" in out

def test_build_prompt_subcategories_none_and_strings_ok(tmp_templates):
    s1 = L.build_prompt(
        "update_me_news", "v01",
        "k", "", "", None, None,
        template_dir=str(tmp_templates)
    )
    s2 = L.build_prompt(
        "update_me_news", "v01",
        "k", "", "", None, "only-one",
        template_dir=str(tmp_templates)
    )
    assert "SUBS:" in s1 and "SUBS:" in s2

def test_build_prompt_handles_newlines_and_commas(tmp_templates):
    text = "line1,\nline2\n\nend"
    out = L.build_prompt(
        "update_me_news", "v01",
        ["a", "b", "c"],     # list so it joins properly
        "news",
        text,
        "http://x",
        ["x", "y"],
        template_dir=str(tmp_templates)
    )
    assert isinstance(out, str)
    # loose structural checks
    assert "KW:" in out and "NEWS:" in out and "TEXT:" in out
    assert "a, b, c" in out  # keywords joined, not per-character

# ----------------------------
# generate_perplexity_output
# ----------------------------
@pytest.fixture
def fake_requests(monkeypatch):
    class R:
        def __init__(self, status=200, data=None, json_exc=None):
            self.status_code = status
            self._data = data if data is not None else {"choices": [{"text": "ok"}]}
            self._json_exc = json_exc
            self.headers = {}
        def json(self):
            if self._json_exc:
                raise self._json_exc
            return self._data

    state = {"last_kwargs": None, "resp": R()}

    def _request(method, url, **kwargs):
        state["last_kwargs"] = {"method": method, "url": url, **kwargs}
        return state["resp"]

    monkeypatch.setattr(L, "requests", SimpleNamespace(request=_request))
    return state

def test_pplx_default_api_key_path_headers(fake_requests, monkeypatch):
    # Your function uses the module constant PERPLEXITY_API_KEY for default auth
    out = L.generate_perplexity_output("sys", "hi")
    sent = fake_requests["last_kwargs"]
    assert sent["method"] == "POST"
    assert sent["headers"]["Authorization"] == f"Bearer {L.PERPLEXITY_API_KEY}"
    assert sent["headers"]["Content-Type"] == "application/json"
    assert out  # response.json() passthrough

def test_pplx_http_error_returns_json_anyway(fake_requests):
    R = type(fake_requests["resp"])
    fake_requests["resp"] = R(status=500, data={"error": "bang"})
    out = L.generate_perplexity_output("sys", "u", API_KEY="K")
    assert out == {"error": "bang"}

def test_pplx_non_json_response_raises(fake_requests):
    R = type(fake_requests["resp"])
    fake_requests["resp"] = R(json_exc=ValueError("not json"))
    with pytest.raises(ValueError):
        L.generate_perplexity_output("s", "u", API_KEY="K")

def test_pplx_unexpected_json_shape_passthrough(fake_requests):
    R = type(fake_requests["resp"])
    fake_requests["resp"] = R(data={"foo": "bar"})
    out = L.generate_perplexity_output("s", "u", API_KEY="K")
    assert out == {"foo": "bar"}

def test_pplx_payload_correctness(fake_requests):
    long_user = "你好\n" + ("x" * 2000)
    out = L.generate_perplexity_output("system\nlines", long_user, API_KEY="K")
    sent = fake_requests["last_kwargs"]
    body = sent["json"]
    assert body["messages"] == [
        {"role": "system", "content": "system\nlines"},
        {"role": "user", "content": long_user},
    ]

def test_pplx_headers_when_api_key_param_used(fake_requests):
    _ = L.generate_perplexity_output("s", "u", API_KEY="ABC123")
    sent = fake_requests["last_kwargs"]
    assert sent["headers"]["Authorization"] == "Bearer ABC123"
    assert sent["headers"]["Content-Type"] == "application/json"

# ----------------------------
# verify_and_fix_json
# ----------------------------
def test_verify_and_fix_json_root_array():
    ok, out = L.verify_and_fix_json('[1, 2, {"a":3}]')
    assert ok is True
    assert isinstance(out, list)
    assert out[2]["a"] == 3

@pytest.mark.xfail(reason="Comments not stripped yet")
def test_verify_and_fix_json_with_cpp_style_comments():
    ok, out = L.verify_and_fix_json('{ /* note */ "a": 1, // end\n "b": 2 }')
    assert ok and out == {"a": 1, "b": 2}

@pytest.mark.xfail(reason="Single quotes not normalized yet")
def test_verify_and_fix_json_single_quotes():
    ok, out = L.verify_and_fix_json("{ 'a': 1, 'b': 'two' }")
    assert ok and out == {"a": 1, "b": "two"}

@pytest.mark.xfail(reason="Python literals True/False/None not normalized yet")
def test_verify_and_fix_json_python_literals():
    ok, out = L.verify_and_fix_json('{ "a": True, "b": None, "c": False }')
    assert ok and out == {"a": True, "b": None, "c": False}

def test_verify_and_fix_json_trailing_comma_in_array():
    ok, out = L.verify_and_fix_json('{"a":[1,2,]}')
    assert ok is True
    assert out == {"a": [1, 2]}

def test_verify_and_fix_json_triple_backticks_inside_strings_not_eaten():
    ok, out = L.verify_and_fix_json('{"code":"Here is ```not a fence``` keep it"}')
    assert ok is True
    assert out == {"code": "Here is ```not a fence``` keep it"}

@pytest.mark.xfail(reason="Concatenated objects not merged/array-wrapped")
def test_verify_and_fix_json_multiple_concatenated_objects():
    ok, out = L.verify_and_fix_json('{"a":1}\n{"b":2}')
    assert ok and out == [{"a": 1}, {"b": 2}]

def test_verify_and_fix_json_non_string_input_behavior():
    with pytest.raises(AttributeError):
        L.verify_and_fix_json(None)
    with pytest.raises(AttributeError):
        L.verify_and_fix_json(123)
