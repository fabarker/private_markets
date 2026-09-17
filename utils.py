
# GeneratePECashFlowSub
def generate_cash_flows_liquid_private(self,
                                       current_total_liquid_assets,
                                       initial_vintages,
                                       annual_commitments: np.array,
                                       liquid_portfolio_yearly_returns,
                                       inflation_paths=None,
                                       ann_commitments_in_dollars=False,
                                       wealth_flows=None,
                                       num_years=20,
                                       use_total_mv=False,
                                       ):

    # VintageYear(commitment: float = 0, initial_value = 0, start_age= 0, calls, distributions, returns, shocks)
    # new_vintages = np.tile(Vintage(0, 0, 0), [len(inital_vintages), num_years])

    num_classes = annual_commitments.shape[0]
    assert num_classes == len(initial_vintages), "Error - mismatch between vintages and commitment arrays"
    distributions = np.zeros([num_classes, num_years])
    capital_calls = np.zeros([num_classes, num_years])
    pe_alloc = np.zeros([num_classes, num_years])
    vintage_exposures = np.zeros([num_years, num_years, num_classes])
    liquid_assets_EOY = np.zeros(num_years)
    total_MV = np.zeros(num_years)


    new_vintages = [
        [Vintage(x, self._schema, 0) for _ in range(num_years)]
        for x in initial_vintages.keys()
    ]

    # loop through number of years
    for year in range(num_years):

        # Loop through all vintages and compute distributions from previous yaer
        for i, asset in enumerate(initial_vintages):

            # for the vintages we have, get the distributions in year
            distributions[i, year] = np.sum([ vy.get_distribution(year) for vy in initial_vintages.get(asset) ])

            for fund_start_year in range(year + 1):
                distributions[i, year] += new_vintages[i][fund_start_year].get_distribution(year - fund_start_year)

        # Compute the total market value at beginning of the year
        for i, asset in enumerate(initial_vintages):
            pe_alloc[i, year] = np.sum([vy.get_BOY_NAV(year) for vy in initial_vintages.get(asset)])
            vintage_exposures[0, year, i] = np.sum([vy.get_BOY_NAV(year) for vy in initial_vintages.get(asset)])

            for fund_start_year in range(year + 1):
                pe_alloc[i, year] += new_vintages[i][fund_start_year].get_BOY_NAV(year - fund_start_year)
                vintage_exposures[fund_start_year, year, i] =  new_vintages[i][fund_start_year].get_BOY_NAV \
                    (year - fund_start_year)

        total_MV[year] = (current_total_liquid_assets if year == 0 else liquid_assets_EOY[yea r -1]) + sum \
            (pe_alloc[:, year])

        # Apply Inflows and Outflows at the end of the year
        if wealth_flows is not None:
            pass

        flows = np.zeros((20, ))

        # Now grow the liquid assets given their return. The after growth NAV will take care of the flows, recieving distributions
        # On year 0, use initial capital; otherwise, grow previous year’s liquid assets
        liquid_assets_EOY_pre_flows = (
            current_total_liquid_assets
            if year == 0
            else liquid_assets_EOY[year - 1] * (1 + liquid_portfolio_yearly_returns[year])
        )

        liquid_assets_EOY[year] = liquid_assets_EOY_pre_flows + np.sum(distributions[:, year]) + flows[year]

        # now we need to create new commitments
        for i, asset in enumerate(initial_vintages):

            # Determine the new vintage commitment amount
            vintage_commit = (
                annual_commitments[i, year]
                if ann_commitments_in_dollars
                else annual_commitments[i, year] * (
                    total_MV[year] if use_total_mv else liquid_assets_EOY[year]
                )
            )

            # construct new vintage year
            new_vintages[i][year] = Vintage(
                new_vintages[i][year].type,
                self._schema,
                vintage_commit,
            )

        # get total capital calls by summing across all vintages
        for i, asset in enumerate(initial_vintages):
            capital_calls[i, year] = np.sum([vy.get_capital_call(year) for vy in initial_vintages.get(asset)])

            for fund_start_year in range(year + 1):
                capital_calls[i, year] += new_vintages[i][fund_start_year].get_capital_call(year - fund_start_year)

            # subtract capital calls from liquid assets
            liquid_assets_EOY[year] -= capital_calls[i, year]

    return PrivateAssetProjections(
        capital_calls,
        distributions,
        initial_vintages.keys(),
        total_MV, pe_alloc,
        annual_commitments,
        vintage_exposures
    )
