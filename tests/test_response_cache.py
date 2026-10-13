import pytest
from pydantic import BaseModel

from llm_long_term_memory.llm.cache import CachingClient, request_key
from llm_long_term_memory.llm.client import Completion


class Verdict(BaseModel):
    answer: str


class Provider:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail
        self.usage = "provider-usage"

    def generate(self, **request):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return Completion(
            text=f"answer {self.calls}",
            model=request["model"],
            input_tokens=10,
            output_tokens=2,
            thinking_tokens=0,
            attempts=1,
            api_latency_ms=12.0,
        )


def ask(client, prompt="Q", **overrides):
    request = dict(role="answerer", model="m", prompt=prompt, system="S", temperature=0.0)
    request.update(overrides)
    return client.generate(**request)


def test_identical_request_is_served_once_and_marked_as_no_call(tmp_path):
    provider = Provider()
    client = CachingClient(provider, tmp_path / "c.db", sample="rep1")
    first, second = ask(client), ask(client)
    assert provider.calls == 1
    assert first.text == second.text == "answer 1"
    assert second.attempts == 0 and second.api_latency_ms == 0.0
    assert (client.hits, client.misses) == (1, 1)
    assert client.usage == "provider-usage"  # other attributes pass through


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "Q with a different context"},
        {"system": "S2"},
        {"model": "m2"},
        {"role": "judge"},
        {"temperature": 0.7},
        {"max_output_tokens": 1024},
        {"schema": Verdict},
    ],
)
def test_any_change_to_the_request_misses(tmp_path, change):
    provider = Provider()
    client = CachingClient(provider, tmp_path / "c.db", sample="rep1")
    ask(client)
    ask(client, **change)
    assert provider.calls == 2


def test_a_second_sample_never_reuses_the_first(tmp_path):
    provider = Provider()
    ask(CachingClient(provider, tmp_path / "c.db", sample="rep1"))
    rep2 = CachingClient(provider, tmp_path / "c.db", sample="rep2")
    assert ask(rep2).text == "answer 2"
    assert provider.calls == 2


def test_scope_separates_requests_the_prompt_does_not_show(tmp_path):
    provider = Provider()
    ask(CachingClient(provider, tmp_path / "c.db", sample="rep1", scope={"store": "a"}))
    ask(CachingClient(provider, tmp_path / "c.db", sample="rep1", scope={"store": "b"}))
    assert provider.calls == 2


def test_estimated_tokens_do_not_change_the_key():
    a = request_key(sample="r", scope=None, model="m", prompt="p", est_input_tokens=1)
    b = request_key(sample="r", scope=None, model="m", prompt="p", est_input_tokens=999)
    assert a == b


def test_failures_are_not_cached_and_a_sample_label_is_required(tmp_path):
    with pytest.raises(ValueError, match="sample"):
        CachingClient(Provider(), tmp_path / "c.db", sample="")
    failing = CachingClient(Provider(fail=True), tmp_path / "c.db", sample="rep1")
    with pytest.raises(RuntimeError):
        ask(failing)
    working = Provider()
    ask(CachingClient(working, tmp_path / "c.db", sample="rep1"))
    assert working.calls == 1


def test_cache_survives_reopening(tmp_path):
    provider = Provider()
    client = CachingClient(provider, tmp_path / "c.db", sample="rep1")
    ask(client)
    client.close()
    assert ask(CachingClient(provider, tmp_path / "c.db", sample="rep1")).text == "answer 1"
    assert provider.calls == 1


def test_eval_builder_wraps_answerer_and_judge_in_the_cache(tmp_path):
    from llm_long_term_memory import cli
    from llm_long_term_memory.config import Settings

    provider = Provider()
    _, _, runner, judge, _ = cli._build(
        "full_context",
        "configs/fallback.yaml",
        client_override=provider,
        settings_override=Settings(LLTM_STORE_DIR=str(tmp_path)),
        response_cache=str(tmp_path / "cache.db"),
        sample="rep1",
    )
    assert isinstance(runner.client, CachingClient) and runner.client.sample == "rep1"
    assert judge.client is runner.client
    with pytest.raises(ValueError, match="sample"):
        cli._build(
            "full_context",
            "configs/fallback.yaml",
            client_override=provider,
            settings_override=Settings(LLTM_STORE_DIR=str(tmp_path)),
            response_cache=str(tmp_path / "cache.db"),
        )
