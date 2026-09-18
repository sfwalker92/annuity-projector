"""Fixed deferred annuity cashflow projection.

A deliberately small, readable model: one policy cohort, annual timesteps,
decrements for mortality and lapse, and a surrender charge scale.
"""

from dataclasses import dataclass


@dataclass
class Assumptions:
    """Inputs to a projection. One cohort, annual timesteps."""

    premium: float = 100_000.0
    crediting_rate: float = 0.035
    discount_rate: float = 0.045
    mortality_rate: float = 0.008
    lapse_rate: float = 0.06
    surrender_charge: tuple = (0.07, 0.06, 0.05, 0.04, 0.03, 0.0)
    years: int = 10
    market_rate: float = 0.035
    lapse_sensitivity: float = 2.0
    max_lapse_rate: float = 0.35

    def surrender_charge_at(self, year_index: int) -> float:
        """Surrender charge applying in a given projection year (0-based)."""
        if year_index < len(self.surrender_charge):
            return self.surrender_charge[year_index]
        return 0.0

    def lapse_rate_at(self, year_index: int) -> float:
        """Lapse rate for a given projection year (0-based), rising when the
        market rate exceeds the crediting rate on offer."""
        excess = max(0.0, self.market_rate - self.crediting_rate)
        rate = self.lapse_rate + self.lapse_sensitivity * excess
        return min(rate, self.max_lapse_rate)


@dataclass
class YearResult:
    """One projection year, all amounts per unit of initial policy count."""

    year: int
    inforce_start: float
    account_value: float
    death_outgo: float
    surrender_outgo: float
    inforce_end: float


def project(a: Assumptions) -> list[YearResult]:
    """Project the cohort forward, returning one YearResult per year."""
    results = []
    inforce = 1.0
    account_value = a.premium

    for t in range(a.years):
        inforce_start = inforce
        account_value = account_value * (1 + a.crediting_rate)

        deaths = inforce_start * a.mortality_rate
        death_outgo = deaths * account_value

        survivors = inforce_start - deaths
        lapses = survivors * a.lapse_rate_at(t)
        surrender_value = account_value * (1 - a.surrender_charge_at(t))
        surrender_outgo = lapses * surrender_value

        inforce = survivors - lapses

        results.append(
            YearResult(
                year=t + 1,
                inforce_start=inforce_start,
                account_value=account_value,
                death_outgo=death_outgo,
                surrender_outgo=surrender_outgo,
                inforce_end=inforce,
            )
        )

    return results


def present_value(results: list[YearResult], discount_rate: float) -> float:
    """PV of death and surrender outgo, discounted from the end of each year."""
    return sum(
        (r.death_outgo + r.surrender_outgo) / (1 + discount_rate) ** r.year
        for r in results
    )


if __name__ == "__main__":
    assumptions = Assumptions()
    results = project(assumptions)

    print(f"{'Yr':>3} {'Inforce':>9} {'AV':>12} {'Death':>12} {'Surr':>12}")
    for r in results:
        print(
            f"{r.year:>3} {r.inforce_end:>9.4f} {r.account_value:>12,.0f} "
            f"{r.death_outgo:>12,.0f} {r.surrender_outgo:>12,.0f}"
        )

    pv = present_value(results, assumptions.discount_rate)
    print(f"\nPV of outgo: {pv:,.0f}")