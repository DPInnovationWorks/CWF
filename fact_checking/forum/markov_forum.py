"""
Markov 辩论流程编排：
- S0: 初始判断（使用 Leader 直接基于 claim + evidences 判断）
- S1: Trust -> Skeptic -> Leader
- S2: Skeptic -> Trust -> Leader
"""

from typing import Any, Dict, List

from agents.markov.leader_agent import LeaderAgent
from agents.markov.skeptic_agent import SkepticAgent
from agents.markov.trust_agent import TrustAgent
from config import load_env
from config.llm_factory import Provider


class MarkovForum:
    """按照状态转移规则执行多轮辩论"""

    def __init__(
        self,
        min_debate_rounds: int = 1,
        max_debate_rounds: int = 4,
        provider: Provider = "deepseek",
        model: str = "deepseek-chat",
    ):
        load_env()
        self.min_debate_rounds = min_debate_rounds
        self.max_debate_rounds = max_debate_rounds

        self.trust_agent = TrustAgent(provider=provider, model=model)
        self.skeptic_agent = SkepticAgent(provider=provider, model=model)
        self.leader_agent = LeaderAgent(provider=provider, model=model)

    @staticmethod
    def _all_consensus(round_outputs: List[Dict[str, Any]]) -> bool:
        """判断一轮中三方是否达成一致（使用 stance）。"""
        stances = [str(item.get("stance", "0")) for item in round_outputs]
        return len(set(stances)) == 1

    def verify(self, claim: str, print_debug: bool = False) -> Dict[str, Any]:
        """
        执行 Markov 辩论验证。

        返回：
        {
          "final": Leader 最终输出,
          "initial": S0 输出,
          "rounds": 每轮完整轨迹,
          "total_rounds": 实际轮数
        }
        """
        # S0: 初始判断（使用与 Leader 类似的裁决逻辑）
        if print_debug:
          print("====== S0: Leader 初判 ======")
        initial_opinion = self.leader_agent.respond(
            claim=claim,
            previous_opinions=[],
            debate_round=0,
        )
        # 仅第一轮讨论 initial_opinion；之后始终讨论上一轮 leader_out
        previous_opinion_for_round = initial_opinion
        previous_judgment = bool(initial_opinion.get("factuality", False))
        current_state = "S2" if previous_judgment else "S1"
        if print_debug:
          print(f"leader opinion: {initial_opinion.get('opinion')}")
          print(f"====== Leader 初判为 {previous_judgment}，状态转移 S0 -> {current_state} ======")

        rounds: List[Dict[str, Any]] = []
        final_leader = initial_opinion

        for round_idx in range(1, self.max_debate_rounds + 1):
            if current_state == "S1":
                if print_debug:
                    print(f"====== S1: Trust -> Skeptic -> Leader ======")
                    print(f"previous opinion: {previous_opinion_for_round.get('opinion')}")
                trust_out = self.trust_agent.respond(
                    claim,
                    previous_opinion=previous_opinion_for_round,
                    debate_round=round_idx,
                )
                if print_debug:
                    print(f"trust opinion: {trust_out.get('opinion')}")
                skeptic_out = self.skeptic_agent.respond(
                    claim, previous_opinion=trust_out, debate_round=round_idx
                )
                if print_debug:
                    print(f"skeptic opinion: {skeptic_out.get('opinion')}")
                leader_out = self.leader_agent.respond(
                    claim,
                    previous_opinions=[trust_out, skeptic_out],
                    debate_round=round_idx,
                )
                if print_debug:
                    print(f"leader opinion: {leader_out.get('opinion')}")
                seq = [
                    {"agent": "trust", "output": trust_out},
                    {"agent": "skeptic", "output": skeptic_out},
                    {"agent": "leader", "output": leader_out},
                ]
            else:
                if print_debug:
                    print(f"====== S2: Skeptic -> Trust -> Leader ======")
                    print(f"previous opinion: {previous_opinion_for_round.get('opinion')}")
                skeptic_out = self.skeptic_agent.respond(
                    claim,
                    previous_opinion=previous_opinion_for_round,
                    debate_round=round_idx,
                )
                if print_debug:
                    print(f"skeptic opinion: {skeptic_out.get('opinion')}")
                trust_out = self.trust_agent.respond(
                    claim, previous_opinion=skeptic_out, debate_round=round_idx
                )
                if print_debug:
                    print(f"trust opinion: {trust_out.get('opinion')}")
                leader_out = self.leader_agent.respond(
                    claim,
                    previous_opinions=[skeptic_out, trust_out],
                    debate_round=round_idx,
                )
                if print_debug:
                    print(f"leader opinion: {leader_out.get('opinion')}")
                seq = [
                    {"agent": "skeptic", "output": skeptic_out},
                    {"agent": "trust", "output": trust_out},
                    {"agent": "leader", "output": leader_out},
                ]

            round_outputs = [seq[0]["output"], seq[1]["output"], seq[2]["output"]]
            rounds.append({"round": round_idx, "state": current_state, "sequence": seq})

            final_leader = leader_out

            # Step 4: 终止检查
            if round_idx >= self.min_debate_rounds and self._all_consensus(round_outputs):
                break

            # 状态转移：基于 leader 本轮判断
            previous_judgment = bool(leader_out.get("factuality", False))
            current_state = "S2" if previous_judgment else "S1"
            # 下一轮的前序输入更新为 leader 输出
            previous_opinion_for_round = leader_out

        return {
            "initial": initial_opinion,
            "final": final_leader,
            "rounds": rounds,
            "total_rounds": len(rounds),
        }
