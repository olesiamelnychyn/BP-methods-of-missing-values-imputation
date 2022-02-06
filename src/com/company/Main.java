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

	public static void main (String[] args) throws IOException {

		Input input = readInput();
		HybridMethod imputationMethods = new HybridMethod(input.columnPredictors, input.datasetComplete, input.datasetMissing, input.printOnlyFinal);

		imputationMethods.runImputation(input.columnPredicted);
		writeOutput(input.datasetMissing);
	}

	/**
	 * Class which contains all data that user writes as input
	 */
	private static class Input {
		public SimpleDataSet datasetComplete;
		public SimpleDataSet datasetMissing;
		public int columnPredicted;
		public int[] columnPredictors;
		public boolean printOnlyFinal;
	}

	private static Input readInput () throws IOException {

		Input input = new Input();
		Scanner scanner = new Scanner(System.in);

		//file which does not contain missing values and will be used for evaluation
		System.out.println("Enter filename with complete dataset (type \"1\" to use test filename, type \"2\" to use test filename for multiple, skip if there is not one)");
		String in = scanner.nextLine();
		if ("1".equals(in)) {
			input.datasetComplete = DatasetManipulation.readDataset("src/com/company/data/combined_complete.csv", false, false);
		} else if ("2".equals(in)) {
			input.datasetComplete = DatasetManipulation.readDataset("src/com/company/data/combined_multiple_complete.csv", false, false);
		} else if ("".equals(in) || in == null) {
			input.datasetComplete = null;
		} else {
			input.datasetComplete = DatasetManipulation.readDataset(in, false, false);
		}

		//boolean which controls if 0 should be taken as missing value
		System.out.println("Should 0 be taken as missing value? (0 - yes, any other key - no)");
		boolean isZeroMissing = "0".equals(scanner.nextLine());

		//file which contains missing values
		System.out.println("Enter filename with incomplete dataset (type \"1\" to use test filename, type \"2\" to use test filename for multiple)");
		in = scanner.nextLine();
		if ("1".equals(in)) {
			input.datasetMissing = DatasetManipulation.readDataset("src/com/company/data/combined_missing.csv", true, isZeroMissing);
		} else if ("2".equals(in)) {
			input.datasetMissing = DatasetManipulation.readDataset("src/com/company/data/combined_multiple_missing.csv", true, isZeroMissing);
		} else {
			input.datasetMissing = DatasetManipulation.readDataset(scanner.nextLine(), true, isZeroMissing);
		}

		//index(es) of column(s) to be predicted
		System.out.println("Enter index of the column to be predicted (starting from 0) (type -1 to impute all dependent values)");
		input.columnPredicted = Integer.parseInt(scanner.nextLine());
		if (input.columnPredicted >= input.datasetComplete.getDataMatrix().cols()) {
			throw new IndexOutOfBoundsException("Index of predicted column is out of range");
		}

		//index(es_ of column(s) to be used for predicting
		System.out.println("Enter index(es) of predictor(s) (starting from 0)"); // src/com/company/data/combined_missing.csv
		String[] predictors = scanner.nextLine().split(" ", -1);
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

		//boolean which controls the amount of printing out to the output
		System.out.println("Print only final measures? (0 - no, any other key - yes)");
		in = scanner.nextLine();
		input.printOnlyFinal = !"0".equals(in);

		return input;
	}

	private static void writeOutput (SimpleDataSet dataset) throws IOException {
		Scanner scanner = new Scanner(System.in);
		//file where to write dataset with imputed values
		System.out.println("Enter filename of the output file (type \"1\" to use test filename, type \"0\" to skip saving)");
		String in = scanner.nextLine();
		if ("1".equals(in)) {
			CSV.write(dataset, Paths.get("src/com/company/data/imputed.csv"), ',');
		} else if ("0".equals(in) || in.isEmpty()) {
			return;
		} else {
			CSV.write(dataset, Paths.get(scanner.nextLine()), ',');
		}
	}
}
