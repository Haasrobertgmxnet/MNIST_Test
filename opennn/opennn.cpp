// opennn.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <print>
#include "opennn.h"
#include "opennn_strings.h"
#include "PathNameService.h"
#include "Timer.h"

using namespace opennn;

int main()
{
    Helper::Timer tim;

    // Define the problem
    auto pathName = std::string{};
    pathName = Helper::PathNameService::findFileAboveCurrentDirectory("mnist_images").value();
    auto data_set = DataSet{};
    data_set.set_data_file_name(pathName);
    data_set.read_bmp();
    data_set.scale_input_variables();

    const Index input_variables_number = data_set.get_input_variables_number();
    const Index target_variables_number = data_set.get_target_variables_number();

    constexpr auto hidden_neurons_number1 = Index{ 200 };
    constexpr auto hidden_neurons_number2 = Index{ 100 };
    constexpr auto maximum_epochs_number{ 50 };

    NeuralNetwork neural_network_(NeuralNetwork::ProjectType::Classification, { 784, 200, 100, 10 });

    //auto training_strategy = TrainingStrategy(&neural_network, &data_set);
    //training_strategy.set_maximum_epochs_number(maximum_epochs_number);
    //training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
    //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    //training_strategy.perform_training();



    {
        auto pathName = std::string{};
        pathName = Helper::PathNameService::findFileAboveCurrentDirectory("iris_plant_original.csv").value();
        DataSet data_set(pathName, ';', true);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        const Index hidden_neurons_number = 3;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, { input_variables_number, hidden_neurons_number, target_variables_number });

        TrainingStrategy training_strategy(&neural_network, &data_set);

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

        inputs.setValues({{type(5.1),type(3.5),type(1.4),type(0.2)},
                            {type(6.4),type(3.2),type(4.5),type(1.5)},
                            {type(6.3),type(2.7),type(4.9),type(1.8)}});


        outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

        cout << "\nInputs:\n" << inputs << endl;

        cout << "\nOutputs:\n" << outputs << endl;

        cout << "\nConfusion matrix:\n" << confusion << endl;

    }

    std::println("Hello from OpenNN!");
    
}

// Programm ausführen: STRG+F5 oder Menüeintrag "Debuggen" > "Starten ohne Debuggen starten"
// Programm debuggen: F5 oder "Debuggen" > Menü "Debuggen starten"

// Tipps für den Einstieg: 
//   1. Verwenden Sie das Projektmappen-Explorer-Fenster zum Hinzufügen/Verwalten von Dateien.
//   2. Verwenden Sie das Team Explorer-Fenster zum Herstellen einer Verbindung mit der Quellcodeverwaltung.
//   3. Verwenden Sie das Ausgabefenster, um die Buildausgabe und andere Nachrichten anzuzeigen.
//   4. Verwenden Sie das Fenster "Fehlerliste", um Fehler anzuzeigen.
//   5. Wechseln Sie zu "Projekt" > "Neues Element hinzufügen", um neue Codedateien zu erstellen, bzw. zu "Projekt" > "Vorhandenes Element hinzufügen", um dem Projekt vorhandene Codedateien hinzuzufügen.
//   6. Um dieses Projekt später erneut zu öffnen, wechseln Sie zu "Datei" > "Öffnen" > "Projekt", und wählen Sie die SLN-Datei aus.
