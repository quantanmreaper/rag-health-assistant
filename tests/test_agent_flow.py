"""
Integration tests for the Agent chat flow and safety triage.
"""

from rag_health_assistant.agent.assistant import get_health_agent


def test_agent_guideline_retrieval_and_chat():
    agent = get_health_agent()
    
    # 1. Guideline question
    res1 = agent.chat("What are the target blood pressure recommendations for adults with diabetes?")
    assert res1["status"] in ["success", "fallback"]
    assert len(res1["response"]) > 0
    print("--- Test 1 (Guideline question) Passed ---")
    print("Response snippet:", res1["response"][:200], "...\n")

    # 2. Emergency query
    res2 = agent.chat("I have acute crushing chest pain radiating to my left jaw and shortness of breath")
    assert res2["emergency"]["is_emergency"] is True
    assert "Heart Attack" in res2["emergency"]["matched_flags"][0]
    print("--- Test 2 (Emergency Red-Flag Guardrail) Passed ---")
    print("Matched Emergency Flags:", res2["emergency"]["matched_flags"])
    print("Alert message:", res2["emergency"]["alert_message"].encode("ascii", "replace").decode("ascii"), "\n")


if __name__ == "__main__":
    test_agent_guideline_retrieval_and_chat()
    print("All agent flow tests passed successfully!")
