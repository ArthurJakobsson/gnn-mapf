﻿/* Copyright (C) Jiaoyang Li
* Unauthorized copying of this file, via any medium is strictly prohibited
* Confidential
* Written by Jiaoyang Li <jiaoyanl@usc.edu>, May 2020
*/

/*driver.cpp
* Solve a MAPF instance on 2D grids.
*/
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include "ECBS.h"
#include <fstream>
#include <boost/filesystem.hpp>
#include <string.h>

// TODO delete
void print(vector <vector<int>> & a) {
   std::cout << "The vector elements are : \n";
   
  for(int i=0; i < a.size(); i++){
      for(int j = 0; j < a.at(i).size(); j++){
          cout << (a.at(i)).at(j) << " ";
      }
	  cout << "\n";
      cout << "________________________________________________________________________________________________________________\n";
  }
}

/* Main function */
int main(int argc, char** argv)
{
	namespace po = boost::program_options;
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")

		// params for the input instance and experiment settings
		("map,m", po::value<string>()->required(), "input file for map")
		("agents,a", po::value<string>()->required(), "input file for agents")
		// ("scenname", po::value<string>()->required(), "input scenname for agents, used for saving bds")
		("output,o", po::value<string>(), "output file for statistics")
		("outputPaths", po::value<string>(), "output file for paths")
		("agentNum,k", po::value<int>()->default_value(0), "number of agents")
		("cutoffTime,t", po::value<double>()->default_value(7200), "cutoff time (seconds)")
		("screen,s", po::value<int>()->default_value(1), "screen option (0: none; 1: results; 2:all)")
		("stats", po::value<bool>()->default_value(false), "write to files some detailed statistics")
		// ("batchFolder", po::value<string>()->default_value(""), "Folder to save outputs") // NEW: batch output folder
		("seed", po::value<int>()->default_value(5), "seed for tiebreaker low-level node selection") // NEW: seed
		("bd_filename", po::value<string>(), "filepath for saving bds")
		("firstIter", po::value<bool>()->default_value(false), "choose whether to output bds")

		// params for CBS node selection strategies
		("highLevelSolver", po::value<string>()->default_value("EES"), "the high-level solver (A*, A*eps, EES, NEW)")
		("lowLevelSolver", po::value<bool>()->default_value(true), "using suboptimal solver in the low level")
		("inadmissibleH", po::value<string>()->default_value("Global"), "inadmissible heuristics (Zero, Global, Path, Local, Conflict)")
		("suboptimality", po::value<double>()->default_value(1.2), "suboptimality bound")

		// params for CBS improvement
		("heuristics", po::value<string>()->default_value("WDG"), "admissible heuristics for the high-level search (Zero, CG,DG, WDG)")
		("prioritizingConflicts", po::value<bool>()->default_value(true), "conflict prioirtization. If true, conflictSelection is used as a tie-breaking rule.")
		("bypass", po::value<bool>()->default_value(true), "Bypass1")
		("disjointSplitting", po::value<bool>()->default_value(false), "disjoint splitting")
		("rectangleReasoning", po::value<bool>()->default_value(true), "rectangle reasoning")
		("corridorReasoning", po::value<bool>()->default_value(true), "corridor reasoning")
		("targetReasoning", po::value<bool>()->default_value(true), "target reasoning")
		("sipp", po::value<bool>()->default_value(0), "using SIPPS as the low-level solver")
		("restart", po::value<int>()->default_value(0), "rapid random restart times")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		cout << desc << endl;
		return 1;
	}

	po::notify(vm);
	if (vm["suboptimality"].as<double>() < 1)
	{
		cerr << "Suboptimal bound should be at least 1!" << endl;
		return -1;
	}

	high_level_solver_type s;
	if (vm["highLevelSolver"].as<string>() == "A*")
		s = high_level_solver_type::ASTAR;
	else if (vm["highLevelSolver"].as<string>() == "A*eps")
		s = high_level_solver_type::ASTAREPS;
	else if (vm["highLevelSolver"].as<string>() == "EES")
		s = high_level_solver_type::EES;
	else if (vm["highLevelSolver"].as<string>() == "NEW")
		s = high_level_solver_type::NEW;
	else
	{
		cout << "WRONG high level solver!" << endl;
		return -1;
	}

	if (s == high_level_solver_type::ASTAR && vm["suboptimality"].as<double>() > 1)
	{
		cerr << "A* cannot perform suboptimal search!" << endl;
		return -1;
	}

    heuristics_type h;
	if (vm["heuristics"].as<string>() == "Zero")
		h = heuristics_type::ZERO;
	else if (vm["heuristics"].as<string>() == "CG")
		h = heuristics_type::CG;
	else if (vm["heuristics"].as<string>() == "DG")
		h = heuristics_type::DG;
	else if (vm["heuristics"].as<string>() == "WDG")
		h = heuristics_type::WDG;
	else
	{
		cout << "WRONG heuristics strategy!" << endl;
		return -1;
	}

    if ((h == heuristics_type::CG || h == heuristics_type::DG) && vm["lowLevelSolver"].as<bool>())
    {
        cerr << "CG or DG heuristics do not work with low level of suboptimal search!" << endl;
        return -1;
    }

	heuristics_type h_hat; // inadmissible heuristics
	if (s == high_level_solver_type::ASTAR ||
	    s == high_level_solver_type::ASTAREPS ||
	    vm["inadmissibleH"].as<string>() == "Zero")
		h_hat = heuristics_type::ZERO;
	else if (vm["inadmissibleH"].as<string>() == "Global")
		h_hat = heuristics_type::GLOBAL;
	else if (vm["inadmissibleH"].as<string>() == "Path")
		h_hat = heuristics_type::PATH;
	else if (vm["inadmissibleH"].as<string>() == "Local")
		h_hat = heuristics_type::LOCAL;
	else if (vm["inadmissibleH"].as<string>() == "Conflict")
		h_hat = heuristics_type::CONFLICT;
	else
	{
		cout << "WRONG inadmissible heuristics strategy!" << endl;
		return -1;
	}

	conflict_selection conflict = conflict_selection::EARLIEST;
	node_selection n = node_selection::NODE_CONFLICTPAIRS;

	// NEW: set the seed
	int t = vm["seed"].as<int>();
	srand(t);

	// // NEW: prepare log writing under batchrunner system
	// std::string results_path = "logs/";

	// string all_logs_dir;
	// if (vm["batchFolder"].as<string>() != "") {
	// 	all_logs_dir = results_path + vm["batchFolder"].as<string>();
	// 	boost::filesystem::create_directories(all_logs_dir);
	// }
	// auto saveResultsPath = all_logs_dir + "/" + vm["output"].as<string>();

	///////////////////////////////////////////////////////////////////////////
	// load the instance
	Instance instance(vm["map"].as<string>(), vm["agents"].as<string>(),
		vm["agentNum"].as<int>());

	// srand(0);
	int runs = 1 + vm["restart"].as<int>();
	//////////////////////////////////////////////////////////////////////
    // initialize the solver
	if (vm["lowLevelSolver"].as<bool>())
    {
        ECBS ecbs(instance, vm["sipp"].as<bool>(), vm["screen"].as<int>());
        ecbs.setPrioritizeConflicts(vm["prioritizingConflicts"].as<bool>());
        ecbs.setDisjointSplitting(vm["disjointSplitting"].as<bool>());
        ecbs.setBypass(vm["bypass"].as<bool>());
        ecbs.setRectangleReasoning(vm["rectangleReasoning"].as<bool>());
        ecbs.setCorridorReasoning(vm["corridorReasoning"].as<bool>());
        ecbs.setHeuristicType(h, h_hat);
        ecbs.setTargetReasoning(vm["targetReasoning"].as<bool>());
        ecbs.setMutexReasoning(false);
        ecbs.setConflictSelectionRule(conflict);
        ecbs.setNodeSelectionRule(n);
        ecbs.setSavingStats(vm["stats"].as<bool>());
        ecbs.setHighLevelSolver(s, vm["suboptimality"].as<double>());

        //////////////////////////////////////////////////////////////////////
		// NEW: write bd info to file

		// print(ecbs.bds);
		// cout << ecbs.bds.size() << "\n";

		// Only make bd files if it is the first iteration
		if(vm["firstIter"].as<bool>())
		{
			ofstream file;
			// string scen = vm["scenname"].as<string>();//.substr(17);
			// string bd_file = "./data_collection/eecbs/raw_data/bd/" + scen + std::to_string(vm["agentNum"].as<int>()) + ".txt";
			string bd_file = vm["bd_filename"].as<string>();
			// cout << bd_file;
			file.open(bd_file);
			// write dimensions,
			file << instance.num_of_rows << "," << instance.num_of_cols << endl;
			// then bd information to file uniquely defined by instance and agent number
			for(int i=0;i<ecbs.bds.size();i++) {
				for(int j = 0; j < ecbs.bds.at(i).size(); j++) {
					file << ecbs.bds.at(i).at(j) << ",";
				}
				file << endl;
			}
			file.close();
		}
        //////////////////////////////////////////////////////////////////////
        // run
        double runtime = 0;
        int lowerbound = 0;
        for (int i = 0; i < runs; i++)
        {
            ecbs.clear();
            ecbs.solve(vm["cutoffTime"].as<double>() / runs, lowerbound);
            runtime += ecbs.runtime;
            if (ecbs.solution_found)
                break;
            lowerbound = ecbs.getLowerBound();
            ecbs.randomRoot = true;
            cout << "Failed to find solutions in Run " << i << endl;
        }
        ecbs.runtime = runtime; 
        if (vm.count("output"))
            ecbs.saveResults(vm["output"].as<string>(), vm);
        if (ecbs.solution_found && vm.count("outputPaths"))
            ecbs.savePaths(vm["outputPaths"].as<string>(), instance.num_of_rows, instance.num_of_cols);
        /*size_t pos = vm["output"].as<string>().rfind('.');      // position of the file extension
        string output_name = vm["output"].as<string>().substr(0, pos);     // get the name without extension
        cbs.saveCT(output_name); // for debug*/
        // if (vm["stats"].as<bool>())
        //     ecbs.saveStats(vm["output"].as<string>(), vm["scenname"].as<string>());
        ecbs.clearSearchEngines();
    }
    else
    {
        CBS cbs(instance, vm["sipp"].as<bool>(), vm["screen"].as<int>());
        cbs.setPrioritizeConflicts(vm["prioritizingConflicts"].as<bool>());
        cbs.setDisjointSplitting(vm["disjointSplitting"].as<bool>());
        cbs.setBypass(vm["bypass"].as<bool>());
        cbs.setRectangleReasoning(vm["rectangleReasoning"].as<bool>());
        cbs.setCorridorReasoning(vm["corridorReasoning"].as<bool>());
        cbs.setHeuristicType(h, h_hat);
        cbs.setTargetReasoning(vm["targetReasoning"].as<bool>());
        cbs.setMutexReasoning(false);
        cbs.setConflictSelectionRule(conflict);
        cbs.setNodeSelectionRule(n);
        cbs.setSavingStats(vm["stats"].as<bool>());
        cbs.setHighLevelSolver(s, vm["suboptimality"].as<double>());
        //////////////////////////////////////////////////////////////////////
        // run
        double runtime = 0;
        int lowerbound = 0;
        for (int i = 0; i < runs; i++)
        {
            cbs.clear();
            cbs.solve(vm["cutoffTime"].as<double>() / runs, lowerbound);
            runtime += cbs.runtime;
            if (cbs.solution_found)
                break;
            lowerbound = cbs.getLowerBound();
            cbs.randomRoot = true;
            cout << "Failed to find solutions in Run " << i << endl;
        }
        cbs.runtime = runtime;
        if (vm.count("output"))
            cbs.saveResults(vm["output"].as<string>(), vm);
        if (cbs.solution_found && vm.count("outputPaths"))
            cbs.savePaths(vm["outputPaths"].as<string>(), instance.num_of_rows, instance.num_of_cols);
        // if (vm["stats"].as<bool>())
        //     cbs.saveStats(vm["output"].as<string>(), vm["scenname"].as<string>());
        cbs.clearSearchEngines();
    }
	return 0;

}