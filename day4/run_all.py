#!/usr/bin/env python3
"""Run all Day 4 module demos."""
import sys
sys.path.insert(0, "src")

def run(name, fn):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print('='*70)
    try:
        fn()
        print(f"  [OK] {name} completed")
    except Exception as e:
        print(f"  [FAIL] {name} failed: {e}")

if __name__ == "__main__":
    from day4.langgraph_basics    import main as m1
    from day4.tools_agents        import main as m2
    from day4.memory_persistence  import main as m3
    from day4.human_in_loop       import main as m4
    from day4.multi_agent         import main as m5
    from day4.mcp_integration     import main as m6

    run("LangGraph Basics",       m1)
    run("Tools & Agents",         m2)
    run("Memory & Persistence",   m3)
    run("Human-in-the-Loop",      m4)
    run("Multi-Agent Systems",    m5)
    run("MCP Integration",        m6)
