#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <random>
#include <fstream>

using precise_t = long double;
using equation_t = std::function<precise_t(std::vector<precise_t> const&)>;
using vector_t = std::vector<precise_t>;
using matrix_t = std::vector<vector_t>;

constexpr static precise_t ATMOSPHERE_PRESSURE = 100'000;
constexpr static precise_t GAZ_R = 8.314;
constexpr static precise_t PRECISION = 1e-12;
constexpr static precise_t EPS = 1e-9;

static inline
precise_t get_Gibbs_impl(std::array<precise_t, 7> const& f,
                         precise_t T,
                         precise_t H298)
{
    auto t = T / 10000; //!< T * multiplier to make formulas work

    precise_t t1 = f[0];
    auto t2 = f[1] * logl(t);
    auto t3 = f[2] / (t * t);
    auto t4 = f[3] / t;
    auto t5 = f[4] * t;
    auto t6 = f[5] * t * t;
    auto t7 = f[6] * t * t * t;

    return H298 - T * (t1 + t2 + t3 + t4 + t5 + t6 + t7);
}

// expected agents:
// HCl, Ga, GaCl, H2, GaCl2, GaCl3,
static inline
precise_t get_Gibbs(char const* agent, precise_t T)
{
    if (!strcmp(agent, "HCl"))
    {
        std::array<precise_t, 7> f{
            243.9878, 23.15984, 0.001819985, 0.6147384, 51.16604, -36.89502,
            9.174252
        };
        return get_Gibbs_impl(f, T, -92310);
    }

    if (!strcmp(agent, "Ga"))
    {
        std::array<precise_t, 7> f{
            125.9597, 26.03107, 0.001178297, 0.13976, -0.5698425, 0.04723008,
            7.212525
        };
        return get_Gibbs_impl(f, T, 0);
    }

    if (!strcmp(agent, "H2"))
    {
        std::array<precise_t, 7> f{
            205.5368, 29.50487, 0.000168424, 0.86065612, -14.95312, 78.18955,
            -82.78981
        };
        return get_Gibbs_impl(f, T, 0);
    }

    if (!strcmp(agent, "GaCl"))
    {
        std::array<precise_t, 7> f{
            332.2718, 37.11052, -0.000746187, 1.1606512, 4.891346, -4.467591,
            5.506236
        };
        return get_Gibbs_impl(f, T, -70553);
    }

    if (!strcmp(agent, "GaCl2"))
    {
        std::array<precise_t, 7> f{
            443.2976, 57.745845, -0.002265112, 1.8755545, 3.66186, -9.356338,
            15.88245
        };
        return get_Gibbs_impl(f, T, -241238);
    }

    if (!strcmp(agent, "GaCl3"))
    {
        std::array<precise_t, 7> f{
            526.8113, 82.03355, -0.003486473, 2.6855923, 8.278878, -14.5678,
            12.8899
        };
        return get_Gibbs_impl(f, T, -431573);
    }

    assert(false && "Unrecognized agent");
    throw std::runtime_error(
        "get_Gibbs on unrecognized agent: " + std::string(agent)
    );
}

static inline
precise_t get_temp_constant(std::string const& react, precise_t T)
{
    if (react == "2HCl+2Ga=2GaCl+H2")
    {
        auto c1 = get_Gibbs("HCl", T), c2 = get_Gibbs("Ga", T),
            c3 = get_Gibbs("GaCl", T), c4 = get_Gibbs("H2", T);
        return expl(-2 * (c1 + c2 - c3 - c4 / 2) / (GAZ_R * T))
               / ATMOSPHERE_PRESSURE;
    }

    if (react == "2HCl+Ga=GaCl2+H2")
    {
        auto c1 = get_Gibbs("HCl", T), c2 = get_Gibbs("Ga", T),
            c3 = get_Gibbs("GaCl2", T), c4 = get_Gibbs("H2", T);
        return expl(-(2.0 * c1 + c2 - c3 - c4) / (GAZ_R * T));
    }

    if (react == "6HCl+2Ga=2GaCl3+3H2")
    {
        auto c1 = get_Gibbs("HCl", T), c2 = get_Gibbs("Ga", T),
            c3 = get_Gibbs("GaCl3", T), c4 = get_Gibbs("H2", T);
        return expl(-2 * (3 * c1 + c2 - c3 - 3 * c4 / 2) / (GAZ_R * T))
               * ATMOSPHERE_PRESSURE;
    }
    assert(false && "Unrecognized reaction");
    throw std::runtime_error(
        "get_temp_constant on unrecognized reaction: " + react
    );
}

static inline
precise_t get_pressure(std::string const& agent)
{
    if (agent == "HCl") return 10'000.0;
    if (agent == "N2") return 90'000;
    return 0;
}

static inline
precise_t get_mu(std::string const& agent)
{
    if (agent == "HCl") return 36.461;
    if (agent == "H2") return 2.016;
    if (agent == "N2") return 28.0135;
    if (agent == "GaCl") return 105.173;
    if (agent == "GaCl2") return 140.626;
    if (agent == "GaCl3") return 176.080;
    if (agent == "Ga") return 69.723;

    assert(false && "Unrecognized agent");
    throw std::runtime_error("get_mu on unrecognized agent: " + agent);
}

static inline
// expected agents: "HCl", "GaCl", "GaCl2", "GaCl3", "H2", "N2"
precise_t get_sigma(std::string const& agent)
{
    if (agent == "HCl") return 2.737;
    if (agent == "H2") return 2.93;
    if (agent == "N2") return 3.798;
    if (agent == "GaCl") return 3.696;
    if (agent == "GaCl2") return 4.293;
    if (agent == "GaCl3") return 5.034;

    assert(false && "Unrecognized agent");
    throw std::runtime_error("get_sigma on unrecognized agent: " + agent);
}

static inline
precise_t get_epsil(std::string const& agent)
{
    if (agent == "HCl") return 167.1;
    if (agent == "H2") return 34.1;
    if (agent == "N2") return 71.4;
    if (agent == "GaCl") return 348.2;
    if (agent == "GaCl2") return 465;
    if (agent == "GaCl3") return 548.24;

    assert(false && "Unrecognized agent");
    throw std::runtime_error("get_epsil on unrecognized agent: " + agent);
}

static inline
precise_t get_diffusion(std::string const& agent, precise_t T)
{
    auto const agent_sigma = get_sigma(agent);
    auto const agent_epsil = get_epsil(agent);
    auto const agent_mu = get_mu(agent);

    auto const N2_sigma = get_sigma("N2");
    auto const N2_epsil = get_epsil("N2");
    auto const N2_mu = get_mu("N2");

    precise_t epsilon = sqrtl(agent_epsil * N2_epsil);
    precise_t sigma = (agent_sigma + N2_sigma) / 2;
    precise_t omega = 1.074 * powl(T / epsilon, -0.1604);
    precise_t mu = 2 * agent_mu * N2_mu / (agent_mu + N2_mu);

    auto press = ATMOSPHERE_PRESSURE;
    return 0.02628 * powl(T, 1.5) / (press * sigma * omega * sqrtl(mu));
}

template<typename Numeric,
         typename Generator = std::mt19937>
Numeric random(Numeric from, Numeric to)
{
    using distribution_t = typename std::conditional<
        std::is_integral<Numeric>::value,
        std::uniform_int_distribution<Numeric>,
        std::uniform_real_distribution<Numeric>
    >::type;

    thread_local static Generator gen(std::random_device{}());
    thread_local static distribution_t distribute;

    return
        distribute(
            gen,
            typename distribution_t::param_type{
                from, to
            }
        );
}

static inline
precise_t derivative(equation_t const& eq,
                     size_t index,
                     vector_t& point)
{
    auto value = eq(point);
    point[index] += EPS;
    auto result = (eq(point) - value) / EPS;
    point[index] -= EPS;
    return result;
}

static inline
void derivative(equation_t const& eq,
                vector_t input,
                vector_t& result)
{
    result.clear();
    result.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        result[i] = derivative(eq, i, input);
}

static inline
void solve_system(matrix_t& m, vector_t& result)
{
    size_t n = result.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    for (size_t iter = 0; iter < n; ++iter)
    {
        size_t max_row = iter, max_col = iter;
        for (size_t i = iter; i < n; ++i)
        {
            for (size_t j = iter; j < n; ++j)
            {
                if (fabsl(m[i][j]) > fabsl(m[max_row][max_col]))
                {
                    max_row = i;
                    max_col = j;
                }
            }
        }

        std::swap(m[iter], m[max_row]);

        for (size_t i = 0; i < n; i++)
            std::swap(m[i][max_col], m[i][iter]);

        std::swap(order[iter], order[max_col]);
        std::swap(result[iter], result[max_row]);

        for (size_t i = iter + 1; i < n; ++i)
        {
            auto factor = m[i][iter] / m[iter][iter];
            result[i] -= factor * result[iter];
            for (size_t j = iter; j < n; ++j)
                m[i][j] -= factor * m[iter][j];
        }
    }

    vector_t tmp(n);
    for (ptrdiff_t i = n - 1; i >= 0; --i)
    {
        precise_t sum = 0.0;
        for (size_t j = i + 1; j < n; ++j)
            sum += m[i][j] * tmp[j];
        tmp[i] = (result[i] - sum) / m[i][i];
    }

    for (size_t i = 0; i < n; ++i)
        result[order[i]] = tmp[i];
}

static inline
void find_gradient(std::vector<equation_t> const& eqs,
                   vector_t const& input,
                   vector_t& result)
{
    size_t size = eqs.size();
    matrix_t matrix(size);
    result.clear();
    result.resize(size);

    for (size_t i = 0; i < size; ++i)
    {
        auto& eq = eqs[i];
        result[i] = -eq(input);
        derivative(eq, input, matrix[i]);
    }

    solve_system(matrix, result);
}

static inline
auto dist_to_0(std::vector<equation_t> const& eqs,
               vector_t const& point)
{
    precise_t sum = 0;
    for (auto&& eq : eqs)
    {
        auto value_in_point = eq(point);
        sum += value_in_point * value_in_point;
    }
    return sum;
}

static inline
auto dist_to_0(std::vector<equation_t> const& eqs,
               vector_t const& point,
               vector_t const& dx,
               precise_t factor)
{
    vector_t new_point(eqs.size());
    for (size_t i = 0; i < eqs.size(); i++)
        new_point[i] = point[i] + dx[i] * factor;
    return dist_to_0(eqs, new_point);
}

static inline
precise_t find_local_min(std::vector<equation_t> const& eqs,
                         vector_t const& point,
                         vector_t const& dx)
{
    equation_t eq = [&](vector_t const& v)
    {
        return dist_to_0(eqs, point, dx, v[0]);
    };

    precise_t const p = 1;
    precise_t step = 0.5;
    vector_t arg{p};
    auto deriv = derivative(eq, 0, arg);
    precise_t result_point = 1;
    auto value = eq(arg);
    while (step > PRECISION)
    {
        auto dx = result_point - step;
        if (deriv < 0)
            dx = result_point + step;
        arg[0] = dx;
        auto val = eq(arg);
        if (val < value)
        {
            result_point = dx;
            value = val;
            deriv = derivative(eq, 0, arg);
        }
        else
        {
            step /= 2;
        }
    }
    return result_point;
}

static inline
void solve_system(std::vector<equation_t> const& eqs,
                  vector_t& result)
{
    result.clear();
    assert(!eqs.empty() && "Non-empty system expected!");

    auto n = eqs.size();
    result.resize(n);

    for (auto& x : result)
        x = random<precise_t>(0, 1); //is that enough...?

    vector_t dx;
    for (size_t iteration = 0; iteration < 100'000; ++iteration)
    {
        find_gradient(eqs, result, dx);
        auto local_minimum = find_local_min(eqs, result, dx);

        for (size_t j = 0; j < n; j++)
            result[j] += local_minimum * dx[j];

        if (*std::max_element(dx.begin(), dx.end()) < PRECISION)
            break;
    }
}

static inline
std::unordered_map<std::string, precise_t>
single_step(precise_t T, precise_t delta)
{
    std::vector<std::string> const agents{
        "HCl", "GaCl", "GaCl2", "GaCl3", "H2"
    };
    std::vector<std::string> const reactions{
        "2HCl+2Ga=2GaCl+H2", "2HCl+Ga=GaCl2+H2", "6HCl+2Ga=2GaCl3+3H2"
    };

    vector_t temp_coefs;
    vector_t pressure;
    vector_t diffusion;

    for (auto&& react : reactions)
        temp_coefs.emplace_back(get_temp_constant(react, T));

    for (auto&& agent : agents)
    {
        pressure.emplace_back(get_pressure(agent));
        diffusion.emplace_back(get_diffusion(agent, T));
    }

    std::vector<equation_t> eqs{
        [=](vector_t const& v) -> precise_t
        {
            // 2 HCl + 2 Ga = 2 GaCl + H2
            // Pe(HCl)^2 = K4 * Pe(GaCl)^2 * Pe(H2)
            return v[0] * v[0] - temp_coefs[0] * v[1] * v[1] * v[4];
        },
        [=](vector_t const& v) -> precise_t
        {
            // 2 HCl + Ga = GaCl2 + H2
            // Pe(HCl)^2 = K5 * Pe(GaCl2) * Pe(H2)
            return v[0] * v[0] - temp_coefs[1] * v[2] * v[4];
        },
        [=](vector_t const& v) -> precise_t
        {
            // 6 HCl + 2 Ga = 2 GaCl3 + 3 H2
            // Pe(HCl)^6 = K6 * Pe(GaCl3)^2 * Pe(H2)^3
            return powl(v[0], 6) - temp_coefs[2] * v[3] * v[3] * powl(v[4], 3);
        },
        [=](vector_t const& v) -> precise_t
        {
            // G(H) = G(HCl) + 2 * G(H2) = 0
            // D(HCl) * (Pg(HCl) - Pe(HCl)) + 2 * D(H2) * (Pg(H2) - Pe(H2)) = 0
            return diffusion[0] * (pressure[0] - v[0])
                   + 2 * diffusion[4] * (pressure[4] - v[4]);
        },
        [=](vector_t const& v) -> precise_t
        {
            auto result = diffusion[0] * (pressure[0] - v[0]);
            for (int i = 1; i < 4; i++)
                result += i * diffusion[i] * (pressure[i] - v[i]);
            return result;
        }
    };


    vector_t result_solution;
    while (true)
    {
        solve_system(eqs, result_solution);
        assert(result_solution.size() == agents.size()
               && "Smth wrong with system solution: unexpected vector length");
        bool accurate = true;
        for (auto&& x : result_solution)
            accurate &= (x <= ATMOSPHERE_PRESSURE + 1000) && (x >= -1000);
        if (accurate)
        {
            std::cerr << "Got solution for T=" << T << " delta=" << delta
                      << std::endl;
            break;
        }
    }

    std::unordered_map<std::string, precise_t> result;
    result["Temperature"] = T;
    for (size_t i = 0; i < agents.size(); ++i)
        result["Pe(" + agents[i] + ")"] = result_solution[i];

    for (size_t i = 0; i < agents.size(); ++i)
    {
        auto res = diffusion[i] * (pressure[i] - result_solution[i])
                   / (8314 * T * delta);
        result["G(" + agents[i] + ")"] = res;
    }

    precise_t V =
        (result["G(GaCl)"] + result["G(GaCl2)"] + result["G(GaCl3)"])
        * (get_mu("Ga") / 5900.0 /* Density of Ga */ ) * powl(10, 9);
    result["Ve(Ga)"] = V;

    return result;
}

static inline
void make_dump(std::string const& filename,
               std::unordered_map<std::string, precise_t> const& data)
{
    std::ofstream out(filename);

    std::vector<std::string> result_lines;
    result_lines.reserve(data.size());
    for (auto&& rec : data)
        result_lines.emplace_back(
            rec.first + ": " + std::to_string(rec.second)
        );
    std::sort(result_lines.begin(), result_lines.end()); // to make it readable

    for (auto&& line : result_lines)
        out << line << "\n";
    out << std::endl;
    out.flush();
}

static inline
precise_t get_result(std::unordered_map<std::string, precise_t> const& result,
                     std::string const& key)
{
    auto it = result.find(key);
    if (it == result.end())
        throw std::runtime_error(
            "No key \"" + key + "\" presented in the result map"
        );
    return it->second;
}

static inline
auto run()
{
    std::vector<std::unordered_map<std::string, precise_t>> result;

    #pragma omp parallel for
    for (int i = 650; i < 950; i += 5)
    {
        precise_t T = i + 273;
        auto step_result = single_step(T, 0.01);

//        make_dump("GaClx.temp=" + std::to_string(i) + ".yml", step_result);

        #pragma omp critical
        result.emplace_back(std::move(step_result));
    }

    std::sort(
        result.begin(), result.end(),
        [](auto&& lhs, auto&& rhs)
        {
            return get_result(lhs, "Temperature")
                   < get_result(rhs, "Temperature");
        }
    );

    return result;
}

static inline
uint64_t get_hash(std::initializer_list<uint64_t> list)
{
    uint64_t result = 0;
    for (auto&& x : list)
    {
        result *= 116969;
        result += x;
    }
    return result;
}

int main()
{
    int c;
    srand(
        get_hash(
            {
                (uint64_t) time(nullptr), (uint64_t) & main, (uint64_t) & c,
                (uint64_t) & malloc
            }
        ));

    std::ofstream csv1("arrhenius.csv");
    std::ofstream csv2("GaClX.csv");

    csv1 << "1/T,ln(...)" << std::endl;
    csv2 << "1/T,ln(G(GaCl)),ln(G(GaCl2)),ln(G(GaCl3))" << std::endl;

    auto result_data = run();

    for (auto&& run_result : result_data)
    {
        auto speed_val = get_result(run_result, "Ve(Ga)");
        precise_t speed = logl(fabsl(speed_val));
        auto temp_val = 1 / get_result(run_result, "Temperature");
        csv1 << temp_val << "," << speed << "\n";

        auto GaCl = logl(fabsl(get_result(run_result, "G(GaCl)")));
        auto GaCl2 = logl(fabsl(get_result(run_result, "G(GaCl2)")));
        auto GaCl3 = logl(fabsl(get_result(run_result, "G(GaCl3)")));
        csv2 << temp_val << ","
             << GaCl << ","
             << GaCl2 << ","
             << GaCl3 << "\n";
    }

    csv1 << std::endl;
    csv1.flush();
    csv2 << std::endl;
    csv2.flush();
}
