"""Shared SIC -> industry classifier and Finance-exclusion rule.

Single source of truth for the project's industry filter. Every figure
and regression script that needs to bucket firms by industry, or drop
"Finance" firms from a sample, imports from here.

Finance-exclusion rule (committed 2026-05-28):
  Drop SIC codes in:
    - 6000-6099  (Major Group 60: depository institutions)
    - 6200-6299  (Major Group 62: securities brokers/dealers/exchanges)
                 EXCEPT 6211, 6282 -- carved back in (see KEEP set below)
    - 6300-6399  (Major Group 63: insurance carriers)
    - 6400-6499  (Major Group 64: insurance agents/brokers)
                 EXCEPT 6411 -- carved back in
    - 6500-6599  (Major Group 65: real estate)
                 EXCEPT 6512, 6531, 6532 -- carved back in
    - 6700-6799  (Major Group 67: holding/investment offices, incl. REITs + funds)
    - 6111       (federal & federally-sponsored credit agencies: Fannie/Freddie/FHLB)
    - 6159       (misc. federally-sponsored credit agencies)

  KEEP (treated as non-Finance / 'Other', included in analysis):
    Major Group 61 except 6111 and 6159 -- private credit-institution industries:
      6141  Personal credit institutions (Capital One, AmEx, Discover, Synchrony,
            GM Financial, Toyota Motor Credit, SoFi, Affirm, LendingClub, etc.)
      6153  Short-term business credit (Deere Capital, AmEx Credit, Encore, PRA)
      6162  Mortgage bankers (Rocket, PennyMac, Mr. Cooper, Countrywide, IndyMac)
      6163  Loan brokers (LendingTree, Ditech, HFF)
      6172  Finance lessors (Ally Financial, Orix, CIT)
      6189  Asset-backed securities (currently empty in Compustat)
      6199  Finance services NEC (Citigroup parent, others)
    Explicit carve-outs from Major Groups 62 / 64 / 65 (firms that report
    material international reconciliation lines and are professional-services
    / equipment-leasing businesses rather than pure-play Finance; identified
    via systematic scan of reconciliation tables, May 2026):
      6211  Security brokers, dealers, flotation cos (Goldman Sachs, Jefferies,
            Raymond James, Piper Sandler, BGC Group)
      6282  Investment advice (State Street, SEI Investments, Federated Hermes,
            Affiliated Managers, Diamond Hill)
      6411  Insurance agents, brokers and service (Marsh McLennan, Arthur J
            Gallagher, Ryan Specialty, Crawford & Co, Goosehead, Hagerty)
      6512  Operators of apartment buildings -- equipment-leasing / flex-office
            in practice (WillScot Holdings, WeWork, CoreCivic, Target Hospitality)
      6531  Real estate agents and managers -- global commercial-RE services
            (CBRE, Jones Lang LaSalle, Newmark, RE/MAX, Anywhere, Travel+Leisure)
      6532  Real estate dealers (Hilton Grand Vacations, ILG, vacation-ownership
            operators)

Industry-classifier rule (unchanged from the prior per-script copies, except
that the Finance bucket now follows the rule above):
  Pharma/Biotech : SIC 2830-2836 or 8731
  Tech           : SIC 3570-3579, 3670-3679, 7370-7379
  Oil & Gas      : SIC 1310-1389, 2910-2999
  Retail         : SIC 5200-5999
  Manufacturing  : SIC 2000-3999 (excluding the Pharma/Tech ranges above)
  Utilities      : SIC 4900-4939
  Finance        : per `is_finance_sic` above
  Other          : everything else

The Finance check fires *before* the Manufacturing range (so a SIC 6199 firm
gets 'Finance' rather than falling through; same for any overlap).
"""
from __future__ import annotations


# Explicit carve-outs from the broad 6000-6799 Finance range. Firms here
# are professional-services or equipment-leasing businesses with material
# international operations; see module docstring for rationale.
NON_FINANCE_CARVEOUTS = frozenset({6211, 6282, 6411, 6512, 6531, 6532})


def is_finance_sic(sic) -> bool:
    """True if the SIC belongs to the project's Finance-exclusion set."""
    try:
        s = int(sic)
    except (ValueError, TypeError):
        return False
    if s in NON_FINANCE_CARVEOUTS:
        return False
    if 6000 <= s <= 6099:
        return True
    if 6200 <= s <= 6299:
        return True
    if 6300 <= s <= 6399:
        return True
    if 6400 <= s <= 6499:
        return True
    if 6500 <= s <= 6599:
        return True
    if 6700 <= s <= 6799:
        return True
    if s == 6111 or s == 6159:
        return True
    return False


def industry(sic) -> str:
    """Map a SIC code to one of the project's eight industry labels."""
    try:
        s = int(sic)
    except (ValueError, TypeError):
        return "Other"
    if is_finance_sic(s):
        return "Finance"
    if 2830 <= s <= 2836 or s == 8731:
        return "Pharma/Biotech"
    if 3570 <= s <= 3579 or 3670 <= s <= 3679 or 7370 <= s <= 7379:
        return "Tech"
    if 1310 <= s <= 1389 or 2910 <= s <= 2999:
        return "Oil & Gas"
    if 5200 <= s <= 5999:
        return "Retail"
    if 2000 <= s <= 3999:
        return "Manufacturing"
    if 4900 <= s <= 4939:
        return "Utilities"
    return "Other"


# Canonical industry order used by stacked-bar legends, axis labels, etc.
INDUSTRY_ORDER = [
    "Tech",
    "Pharma/Biotech",
    "Manufacturing",
    "Retail",
    "Oil & Gas",
    "Utilities",
    "Finance",
    "Other",
]
