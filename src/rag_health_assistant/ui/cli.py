"""
Interactive Terminal CLI for Diabetes & Hypertension RAG Health Assistant.
"""

import sys
from typing import List, Dict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ..agent.assistant import get_health_agent
from ..tools.bp_classifier import classify_blood_pressure
from ..tools.glucose_analyzer import analyze_blood_glucose, convert_hba1c_to_eag
from ..ingestion.indexer import get_indexed_stats


def print_banner():
    print("=" * 70)
    print("       AURAHEALTH AI - CARDIO-METABOLIC RAG HEALTH ASSISTANT")
    print("   Authoritative Clinical Guidance for Diabetes & Hypertension")
    print("=" * 70)
    stats = get_indexed_stats()
    print(f" Knowledge Base: {stats.get('total_chunks', 0)} guideline chunks indexed in ChromaDB")
    print(" Type 'exit' or 'quit' to close. Type 'help' for available commands.")
    print("=" * 70 + "\n")


def run_cli():
    print_banner()
    agent = get_health_agent()
    history: List[Dict[str, str]] = []

    while True:
        try:
            query = input("\n[Patient/User] > ").strip()
            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("\nTake care of your health! Goodbye.")
                break

            if query.lower() == "help":
                print("\nAvailable Commands:")
                print("  - Type any medical question (e.g. 'What is the DASH diet?')")
                print("  - Type vitals (e.g. 'My BP is 140/90' or 'Fasting sugar 145 mg/dL')")
                print("  - Type 'stats' to view knowledge base status")
                print("  - Type 'exit' to quit\n")
                continue

            if query.lower() == "stats":
                stats = get_indexed_stats()
                print(f"\nKnowledge Base Stats: {stats}\n")
                continue

            print("\nThinking & consulting clinical guidelines...")
            res = agent.chat(user_message=query, chat_history=history[-4:])
            
            if res.get("emergency", {}).get("is_emergency"):
                print("\n" + "!" * 60)
                print("🚨 EMERGENCY ALERT DETECTED:")
                print(res["emergency"].get("alert_message"))
                for proto in res["emergency"].get("action_protocol", []):
                    print(f"  • {proto}")
                print("!" * 60 + "\n")

            if res.get("tools_used"):
                tools_str = ", ".join(t.get("name") for t in res["tools_used"])
                print(f"[Tools Executed: {tools_str}]")

            print(f"\n[AuraHealth AI]:\n{res.get('response')}\n")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": res.get("response")})

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Stay healthy!")
            break


if __name__ == "__main__":
    run_cli()
