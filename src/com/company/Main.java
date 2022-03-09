package com.company;

import com.company.utils.DatasetManipulation;
import jsat.SimpleDataSet;
import jsat.io.CSV;

import java.io.IOException;
import java.io.InvalidObjectException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Scanner;

public class Main {

	static ConfigManager configManager = ConfigManager.getInstance();

	public static void main (String[] args) throws IOException {

		Input input = readInput();
		HybridMethod imputationMethods = new HybridMethod(input.columnPredictors, input.datasetComplete, input.datasetMissing);

		imputationMethods.runImputation(input.columnPredicted);
		writeOutput(input.datasetMissing);
	}

	private static Input readInput () throws IOException {

		boolean useConfig = Boolean.parseBoolean(configManager.get("input.useConfigValues"));
		Input input = new Input();

		String predicted;
		String predictorsString;

		if (useConfig) {
			//file which does not contain missing values and will be used for evaluation
			input.datasetComplete = DatasetManipulation.readDataset(configManager.get("input.completeDataset"), false);
			//file which contains missing values
			input.datasetMissing = DatasetManipulation.readDataset(configManager.get("input.incompleteDataset"), true);
			predicted = configManager.get("input.predicted"); //index(es) of column(s) to be predicted
			predictorsString = configManager.get("input.predictors"); //index(es) of column(s) to be used for predicting

		} else {

			Scanner scanner = new Scanner(System.in);

			//file which does not contain missing values and will be used for evaluation
			System.out.println("Enter filename with complete dataset (type \"1\" to use test filename, skip if there is not one)");
			String in = scanner.nextLine();
			if ("1".equals(in)) {
				input.datasetComplete = DatasetManipulation.readDataset(configManager.get("input.completeDataset"), false);
			} else if ("".equals(in) || in == null) {
				input.datasetComplete = null;
			} else {
				input.datasetComplete = DatasetManipulation.readDataset(in, false);
			}

			//file which contains missing values
			System.out.println("Enter filename with incomplete dataset (type \"1\" to use test filename)");
			in = scanner.nextLine();
			if ("1".equals(in)) {
				input.datasetMissing = DatasetManipulation.readDataset(configManager.get("input.incompleteDataset"), true);
			} else {
				input.datasetMissing = DatasetManipulation.readDataset(scanner.nextLine(), true);
			}

			//index(es) of column(s) to be predicted
			System.out.println("Enter index of the column to be predicted (starting from 0) (type -1 to impute all dependent values)");
			predicted = scanner.nextLine();
			configManager.set("input.predictors", predicted);

			//index(es) of column(s) to be used for predicting
			System.out.println("Enter index(es) of predictor(s) (starting from 0)");
			predictorsString = scanner.nextLine();
			configManager.set("input.predictors", predictorsString);

			// ask whether to rewrite config file with current input
			System.out.println("Store predicted and predictors indexes in config file? y/n");
			if ("y".equals(scanner.nextLine())) {
				configManager.store();
			}
		}

		// parse predicted
		input.columnPredicted = Integer.parseInt(predicted);
		if (input.columnPredicted >= input.datasetComplete.getDataMatrix().cols()) {
			throw new IndexOutOfBoundsException("Index of predicted column is out of range");
		}

		// parse predictors
		String[] predictors = predictorsString.split(" ", -1);
		int nPredictors = predictors.length;
		input.columnPredictors = new int[nPredictors];
		int j = 0;
		for (String predictor : predictors) {
			if (Integer.parseInt(predictor) == input.columnPredicted || Integer.parseInt(predictor) >= input.datasetComplete.getDataMatrix().cols()) {
				continue;
			}
			input.columnPredictors[j++] = Integer.parseInt(predictor);
		}
		if (j == 0) {
			throw new InvalidObjectException("There is no predictors!");
		}
		input.columnPredictors = Arrays.copyOfRange(input.columnPredictors, 0, j);

		return input;
	}

	private static void writeOutput (SimpleDataSet dataset) throws IOException {
		if (!Boolean.parseBoolean(configManager.get("output.skipSaving"))) {
			Scanner scanner = new Scanner(System.in);
			//file where to write dataset with imputed values
			System.out.println("Enter filename of the output file (type \"1\" to use test filename)");
			if ("1".equals(scanner.nextLine())) {
				CSV.write(dataset, Paths.get(configManager.get("output.filename")), ',');
			} else {
				CSV.write(dataset, Paths.get(scanner.nextLine()), ',');
			}
		}
	}

	/**
	 * Class which contains all data that user writes as input
	 */
	private static class Input {
		public SimpleDataSet datasetComplete;
		public SimpleDataSet datasetMissing;
		public int columnPredicted;
		public int[] columnPredictors;
	}
}
