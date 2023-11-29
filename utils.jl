using Plots, LinearAlgebra, Statistics, DataFrames, CSV, NLsolve, Memoize, MLJBase, Pkg, MLJModels, ScikitLearn, Flux, Zygote, Statistics, StatsBase, PyCall, Dates, MLJParticleSwarmOptimization, StatisticalMeasures, MLJTuning, Distributions, MethodChains, SIRUS

ta = pyimport_conda("ta", "ta", "conda-forge")
pd = pyimport_conda("pandas", "pandas", "default")
pandas_datareader = pyimport_conda("pandas_datareader", "pandas-datareader", "anaconda")
pdr = pandas_datareader.data
yf = pyimport_conda("yfinance", "yfinance", "conda-forge")
yf.pdr_override()

py"""
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

def get_data(ticker):
    data = pdr.get_data_yahoo(ticker, start="2000-01-01", end="2023-01-01")
    return data
"""
function get_data(ticker)
    if !isfile("$(ticker)_ta.csv")
        data = py"get_data"("$(ticker)")
        data.to_csv("$(ticker).csv")
        data_ta = ta.add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=true)
        data_ta.to_csv("$(ticker)_ta.csv")
    end
    return DataFrame(CSV.File(open("./$(ticker)_ta.csv"); buffer_in_memory=true))
end


function make_train_test_data(train_tickers, market_tickers, start_date, cut_off)
    # train_datas = get_data.(train_tickers)

    prefix_pairs = [market_tickers[1] => market_tickers[2]; ["" => market_ticker for market_ticker in market_tickers[3:end]]]
    market_data = innerjoin(get_data(market_tickers[1]), get_data(market_tickers[2]), on=:Date, renamecols = prefix_pairs[1])
    for (i, m) in enumerate(get_data.(market_tickers[3:end]))
        market_data = innerjoin(market_data, m, on=:Date, renamecols = prefix_pairs[i+1])
    end
    full_ticker_datas = [innerjoin(train_data, market_data, on=:Date) for train_data in get_data.(train_tickers)]
    train_datas = [full_ticker_data[Date(start_date) .<= full_ticker_data.Date .<= Date(cut_off), :] for full_ticker_data in full_ticker_datas]
    test_datas = [full_ticker_data[Date(cut_off) .<= full_ticker_data.Date, :] for full_ticker_data in full_ticker_datas]
    return train_datas, test_datas
end

function NPMM(data, window)
    np_max = Set()
    np_min = Set()
    N = size(data)[1]
    for i in 1:window:(N - window)
        interval = data.Close[i:(i+window-1)]
        push!(np_max, argmax(interval) + i - 1)
        push!(np_min, argmin(interval) + i - 1)
    end
    targets = 1:N .|> (x -> x in np_max ? 1 : x in np_min ? 0 : NaN)
    return targets
end

function make_NPMM_data(data, window)
    targets = NPMM(data, window)
    data[!, :target] = targets
    return data[.!isnan.(targets), :]
end

function concat_training_datas(train_datas, window)
    reduce(vcat, [make_NPMM_data(train_data, window) for train_data in train_datas])
end

function make_matrices(df)
    feature_set = (DataFrames.select(df, Not([:Date, :target])) |> Matrix)
    target_set = df[:, :target] |> Vector
    return feature_set, target_set
end

function make_test_matrices(df)
    feature_set = (DataFrames.select(df, Not([:Date])) |> Matrix)
    if "target" in names(df)
        return (DataFrames.select(df, Not([:Date, :target])) |> Matrix)
    else
        return feature_set
    end
end


function evaluate_trading_strat(model, test_data, threshold, selected_feature_set)
# model, test_data, threshold, selected_feature_set = mach_selected, test_datas[1], .5, selected_feature_set

    testing = make_test_matrices(test_data)
    signals = MLJBase.predict(model, testing[:, selected_feature_set] |> Matrix) .|> x -> pdf(x, 1)
    # signals = MLJBase.predict(mach, testing |> Matrix) .|> x -> pdf(x, 1)
    Closings = testing[:, 4]
    buys = []
    sells = []
    trades = [[]]
    buy_hold = false
    sell_hold = true
    # threshold = .5
    for (i, v) in enumerate(signals)
        if (v > threshold) && buy_hold
            push!(sells, i)
            push!(trades[end], i)
            buy_hold = false
            sell_hold = true
        else (v < (1 - threshold)) && sell_hold
            push!(buys, i)
            push!(trades, [i])
            buy_hold = true
            sell_hold = false
        end
    end

    buy_set = Set(buys)
    sell_set = Set(sells)
    trades = []
    temp = []
    for i in 1:size(testing)[1]
        if (i in buy_set) && (length(temp) == 0)
            push!(temp, i)
        elseif (i in sell_set) && (length(temp) == 1)
            push!(temp, i)
            push!(trades, Tuple(copy(temp)))
            temp = []
        end
    end

    trade_vals = [Closings[x[2]] - Closings[x[1]] for x in trades]
    # true_trades = [x for x in trades if length(x) > 1]
    total_buy_costs = [Closings[x[1]] for x in trades if length(x) > 1] |> sum
    [
    :win_rate => mean(trade_vals .> 0),
    :mean_val => mean(trade_vals),
    :std_val => std(trade_vals),
    :mean_win => mean(trade_vals[trade_vals .> 0]),
    :mean_loss => mean(trade_vals[trade_vals .< 0]),
    :algo_return => sum(trade_vals),
    :algo_return_rate => mean(trade_vals) * length(trade_vals) / Closings[1],
    :asset_return => (Closings[end] - Closings[1]),
    :return_rate => (Closings[end] - Closings[1]) / Closings[1],
    :trades => trades,
    :trade_vals => trade_vals,
    ]
end

function evaluate_trading_strat_sbc(model, test_data, threshold, selected_feature_set)
    # model, test_data, threshold = mach_sbc, test_datas[1], 0.9
    testing = make_test_matrices(test_data)
    signals = MLJBase.predict(model, DataFrame(testing, :auto))
    signals_high = signals .|> x -> pdf(x, 1)
    signals_low = signals .|> x -> pdf(x, 0)
    signals = signals_high ./ (signals_high .+ signals_low)
    # signals = MLJBase.predict(mach, testing |> Matrix) .|> x -> pdf(x, 1)
    Closings = testing[:, 4]
    buys = []
    sells = []
    trades = [[]]
    buy_hold = false
    sell_hold = true
    # threshold = .5
    for (i, v) in enumerate(signals)
        if (v > threshold) && buy_hold
            push!(sells, i)
            push!(trades[end], i)
            buy_hold = false
            sell_hold = true
        else (v < (1 - threshold)) && sell_hold
            push!(buys, i)
            push!(trades, [i])
            buy_hold = true
            sell_hold = false
        end
    end

    buy_set = Set(buys)
    sell_set = Set(sells)
    trades = []
    temp = []
    for i in 1:size(testing)[1]
        if (i in buy_set) && (length(temp) == 0)
            push!(temp, i)
        elseif (i in sell_set) && (length(temp) == 1)
            push!(temp, i)
            push!(trades, Tuple(copy(temp)))
            temp = []
        end
    end

    trade_vals = [Closings[x[2]] - Closings[x[1]] for x in trades]
    # true_trades = [x for x in trades if length(x) > 1]
    total_buy_costs = [Closings[x[1]] for x in trades if length(x) > 1] |> sum
    [
    :win_rate => mean(trade_vals .> 0),
    :mean_val => mean(trade_vals),
    :std_val => std(trade_vals),
    :mean_win => mean(trade_vals[trade_vals .> 0]),
    :mean_loss => mean(trade_vals[trade_vals .< 0]),
    :algo_return => sum(trade_vals),
    :algo_return_rate => mean(trade_vals) * length(trade_vals) / Closings[1],
    :asset_return => (Closings[end] - Closings[1]),
    :return_rate => (Closings[end] - Closings[1]) / Closings[1],
    :trades => trades,
    :trade_vals => trade_vals,
    ]
end
        