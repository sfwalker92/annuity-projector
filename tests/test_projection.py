"""Tests for the annuity projection.

The point of these is not code coverage. It is to tie the model to
independent calculations, and to check it behaves sensibly when the
assumptions are pushed around.
"""

import pytest

from annuity import Assumptions, project, present_value


def test_year_one_matches_hand_calculation():
    """Year one done by hand: 100,000 at 3.5%, q=0.008, 6% lapse, 7% charge."""
    results = project(Assumptions())
    year_one = results[0]

    assert year_one.account_value == pytest.approx(103_500.0)
    assert year_one.death_outgo == pytest.approx(828.0)
    assert year_one.surrender_outgo == pytest.approx(5_729.0976)
    assert year_one.inforce_end == pytest.approx(0.93248)


def test_no_decrements_leaves_cohort_intact():
    """With no mortality and no lapses, nobody leaves and nothing is paid."""
    a = Assumptions(mortality_rate=0.0, lapse_rate=0.0)
    results = project(a)

    assert all(r.inforce_end == pytest.approx(1.0) for r in results)
    assert results[-1].account_value == pytest.approx(
        a.premium * (1 + a.crediting_rate) ** a.years
    )
    assert present_value(results, a.discount_rate) == pytest.approx(0.0)


def test_inforce_declines_monotonically():
    """In-force should only ever fall, and never turn negative."""
    results = project(Assumptions())

    for r in results:
        assert r.inforce_end < r.inforce_start
        assert r.inforce_end > 0.0


def test_surrender_charge_grades_to_zero():
    """Past the charge period, a surrendering policy gets the full account value."""
    a = Assumptions()
    results = project(a)
    year_seven = results[6]

    deaths = year_seven.inforce_start * a.mortality_rate
    lapses = year_seven.inforce_start - deaths - year_seven.inforce_end

    assert year_seven.surrender_outgo == pytest.approx(
        lapses * year_seven.account_value
    )


def test_higher_crediting_rate_increases_pv_of_outgo():
    """A sensitivity whose sign you can predict before you run it."""
    base = Assumptions()
    richer = Assumptions(crediting_rate=base.crediting_rate + 0.01)

    pv_base = present_value(project(base), base.discount_rate)
    pv_richer = present_value(project(richer), richer.discount_rate)

    assert pv_richer > pv_base

def test_dynamic_lapse_is_inactive_at_default_assumptions():
    """Market rate equals crediting rate, so the dynamic term is zero."""
    a = Assumptions()

    assert a.lapse_rate_at(0) == pytest.approx(a.lapse_rate)


def test_lapses_rise_when_market_beats_crediting():
    """2% behind the market, at sensitivity 2.0, adds 4% to the lapse rate."""
    a = Assumptions(market_rate=0.055)

    assert a.lapse_rate_at(0) == pytest.approx(0.10)

    base_end = project(Assumptions())[-1].inforce_end
    stressed_end = project(a)[-1].inforce_end
    assert stressed_end < base_end


def test_lapse_rate_does_not_fall_below_base():
    """Paying above market does not reduce lapses below the base rate."""
    a = Assumptions(market_rate=0.01)

    assert a.lapse_rate_at(0) == pytest.approx(a.lapse_rate)


def test_lapse_rate_is_capped():
    """However far behind the market, the rate stops at max_lapse_rate."""
    a = Assumptions(market_rate=0.50)

    assert a.lapse_rate_at(0) == pytest.approx(a.max_lapse_rate)