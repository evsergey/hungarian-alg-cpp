#include "hungarian.h"

#include <iostream>
#include <fstream>


int main()
{
	std::vector<double> cost_matrix;
	std::ifstream fs("test.txt");
	if (!fs)
	{
		std::cerr << "File 'test.txt' not found" << std::endl;
		return 1;
	}
	std::vector<size_t> assignment;
	std::vector<size_t> best_assignment;
	bool errors = false;
	for (size_t line = 0; fs; ++line)
	{
		size_t cols, rows;
		fs >> cols >> rows;
		if (!fs)
			break;
		cost_matrix.clear();
		cost_matrix.resize(cols * rows);
		for (double& v : cost_matrix)
			fs >> v;
		double bestCost;
		fs >> bestCost;
		best_assignment.clear();
		best_assignment.resize(rows);
		for (size_t& v : best_assignment)
			fs >> v;
		const double cost = hungarian_alg::solve_with_cost(assignment, std::data(cost_matrix), cols, rows);
		size_t ndims = assignment.size() - std::count(assignment.begin(), assignment.end(), cols);
		if (ndims != std::min(cols, rows))
		{
			std::cerr << "Error at line #" << line << ": ndims=" << ndims << std::endl;
			errors = true;
		}
		else if (std::abs(cost - bestCost) > 1e-9 * static_cast<double>(std::min(rows, cols)))
		{
			std::cerr << "Error at line #" << line << ": best=" << bestCost << ", cost=" << cost << std::endl;
			errors = true;
		}
	}
	return errors ? 2 : 0;
}
