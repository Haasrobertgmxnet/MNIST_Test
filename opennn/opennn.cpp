// opennn.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <print>
#include <exception>
#include "opennn.h"
#include "opennn_strings.h"
#include "PathNameService.h"
#include "Timer.h"

using namespace opennn;

void look(const DataSet& data_set) {

	auto tensor = data_set.get_data();
	auto rows = tensor.dimension(0);  // Erste Dimension (Zeilen)
	auto cols = tensor.dimension(1);  // Zweite Dimension (Spalten)

	int n{0};
	auto vu = data_set.get_variables_uses();
	std::vector<DataSet::VariableUse> vuv{};
	for (auto j = 0; j < cols; ++j) {
		vuv.push_back(vu(j));
		if (vu(j) != DataSet::VariableUse::Input) {
			++n;
		}
	}

	std::vector<std::vector<type>> w{};
	for (auto i = 0; i < rows; ++i) {
		std::vector<type> w0{};
		for (auto j = 0; j < cols; ++j) {
			w0.push_back(tensor(i, j));
		}
		w.push_back(w0);
	}

	std::vector<type> s{};
	for (auto j = cols - n; j < cols; ++j) {
		type s0 = 0;
		for (auto i = 0; i < rows; ++i) {
			s0 += tensor(i, j);
		}
		s.push_back(s0);
	}

	auto ts{ std::accumulate(s.begin(),s.end(),0) };

	std::cout << "Set breakpoint here\n";
}

int main()
{
	{
		auto pathName = std::string{};
		pathName = Helper::PathNameService::findFileAboveCurrentDirectory("iris_plant_original.csv").value();
		DataSet data_set(pathName, ';', true);

		look(data_set);

		const Index input_variables_number = data_set.get_input_variables_number();
		const Index target_variables_number = data_set.get_target_variables_number();

		const Index hidden_neurons_number = 3;

		NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, { input_variables_number, hidden_neurons_number, target_variables_number });

		TrainingStrategy training_strategy(&neural_network, &data_set);
		training_strategy.set_maximum_epochs_number(5);
		training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
		training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
		training_strategy.perform_training();

		// Testing analysis

		const TestingAnalysis testing_analysis(&neural_network, &data_set);

		const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

		Tensor<type, 2> inputs(3, neural_network.get_inputs_number());
		Tensor<type, 2> outputs(3, neural_network.get_outputs_number());

		Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);
		Tensor<Index, 1> outputs_dimensions = get_dimensions(outputs);

		inputs.setValues({ {type(5.1),type(3.5),type(1.4),type(0.2)},
							{type(6.4),type(3.2),type(4.5),type(1.5)},
							{type(6.3),type(2.7),type(4.9),type(1.8)} });


		outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

		cout << "\nInputs:\n" << inputs << endl;

		cout << "\nOutputs:\n" << outputs << endl;

		cout << "\nConfusion matrix:\n" << confusion << endl;

	}

	Helper::Timer tim;

	// Define the problem
	// Declarations for repeated use
	auto pathName = std::string{};
	auto data_set = DataSet{};

	pathName = Helper::PathNameService::findFileAboveCurrentDirectory("mnist_train.csv").value();
	auto raw_data = Eigen::Tensor<type, 2>{ Helper::readCSVToTensor2D<type>(pathName) };

	auto raw_data2 = Eigen::Tensor<type, 2>(raw_data.dimension(0), raw_data.dimension(1));
	int rows = raw_data2.dimension(0);  // Erste Dimension (Zeilen)
	int cols = raw_data2.dimension(1);  // Zweite Dimension (Spalten)

	// Kopiere Spalten 1 bis cols-1
	for (int j = 1; j < cols; ++j)
		for (int i = 0; i < rows; ++i)
			raw_data2(i, j - 1) = raw_data(i, j);

	// Kopiere die erste Spalte ans Ende
	for (int i = 0; i < rows; ++i)
		raw_data2(i, cols - 1) = raw_data(i, 0);

	auto train_data{ Helper::createOneHotCoding(raw_data2) };

	rows = train_data.dimension(0);  // Erste Dimension (Zeilen)
	cols = train_data.dimension(1);  // Zweite Dimension (Spalten)

	data_set = DataSet(train_data);

	// Eigen::Tensor<DataSet::VariableUse, 1> columns_uses(1, 794);
	Eigen::Tensor<DataSet::VariableUse, 1> uses(794);
	for (auto j = 0; j < 784; ++j) {
		uses(j) = DataSet::VariableUse::Input;
	}
	for (auto j = 784; j < 794; ++j) {
		uses(j) = DataSet::VariableUse::Target;
	}
	data_set.set_columns_uses(uses);

	look(data_set);
	data_set.scale_input_variables();
	look(data_set);

	const Index input_variables_number = data_set.get_input_variables_number();
	const Index target_variables_number = data_set.get_target_variables_number();
	constexpr auto hidden_neurons_number1 = Index{ 200 };
	constexpr auto hidden_neurons_number2 = Index{ 100 };
	constexpr auto maximum_epochs_number{ 5 };

	NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, { input_variables_number, hidden_neurons_number1, hidden_neurons_number2, target_variables_number });

	auto training_strategy = TrainingStrategy(&neural_network, &data_set);
	training_strategy.set_maximum_epochs_number(maximum_epochs_number);
	training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
	training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
	try {
		training_strategy.perform_training();
	}
	catch (std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}

	std::cout << "Hello from OpenNN!\n";
	return 0;
}
