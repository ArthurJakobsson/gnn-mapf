/*
 * post processing, e.g., calculating solution quality
 */
#pragma once
#include "dist_table.hpp"
#include "instance.hpp"
#include "metrics.hpp"
#include "utils.hpp"

bool is_feasible_solution(const Instance &ins, const Solution &solution,
                          const int verbose = 0);
void print_stats(const int verbose, const Deadline *deadline,
                 const Instance &ins, const Solution &solution,
                 const double comp_time_ms);
void make_log(const Instance &ins, const Solution &solution,
              const std::string &output_name, const double comp_time_ms,
              const std::string &map_name, const int seed,
              const bool log_short = false  // true -> paths not appear
);

void make_log2(const Instance& ins, const AllSolution& all_solution,
              const std::string& output_name, const std::string& output_agent_paths,
              const double comp_time_ms,
              const std::string& map_name, const std::string& scen_name, const int seed,
              const bool log_short);
