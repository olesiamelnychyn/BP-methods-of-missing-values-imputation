package com.company;

import com.company.utils.DatasetManipulation;
import com.company.utils.ImputationMethods;
import jsat.SimpleDataSet;
import jsat.io.CSV;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Scanner;

public class Main {

	public static void main (String[] args) throws IOException {

		Input input = readInput();
		ImputationMethods imputationMethods = new ImputationMethods(input.columnPredicted, input.columnPredictors, input.datasetComplete, input.datasetMissing);

		String str = "Results:\n";
		System.out.println("\n\n" + str);
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt"));
		writer.write(str);
		writer.close();

		imputationMethods.runImputation(input.columnPredicted);
		writeOutput(input.datasetMissing);
	}

	private static class Input {
		public SimpleDataSet datasetComplete;
		public SimpleDataSet datasetMissing;
		public int columnPredicted;
		public int[] columnPredictors;
	}

	private static Input readInput () throws IOException {

		Input input = new Input();
		Scanner scanner = new Scanner(System.in);

		System.out.println("Enter first filename (type \"1\" to use test filename)");
		String in = scanner.nextLine();

		if ("1".equals(in)) {
			input.datasetComplete = DatasetManipulation.readDataset("src/com/company/data/combined_csv_new.csv", false);
		} else if (in == "" || in == null) {
			input.datasetComplete = null;
		} else {
			input.datasetComplete = DatasetManipulation.readDataset(in, false);
		}

		System.out.println("Enter second filename (type \"1\" to use test filename)");
		in = scanner.nextLine();
		if ("1".equals(in)) {
			input.datasetMissing = DatasetManipulation.readDataset("src/com/company/data/combined_csv_new_miss.csv", true);
		} else {
			input.datasetMissing = DatasetManipulation.readDataset(scanner.nextLine(), true);
		}

		System.out.println("Enter index of the column to be predicted (starting from 0) (type -1 to impute all dependent values)");
		input.columnPredicted = Integer.valueOf(scanner.nextLine());
		if (input.columnPredicted >= input.datasetComplete.getDataMatrix().cols()) {
			throw new IndexOutOfBoundsException("Index of predicted column is out of range");
		}

		System.out.println("Enter index(es) of predictor(s) (starting from 0)"); // src/com/company/data/combined_csv_new_miss.csv
		String[] predictors = scanner.nextLine().split(" ", -1);
		int nPredictors = predictors.length;
		input.columnPredictors = new int[nPredictors];
		int j = 0;
		for (int i = 0; i < nPredictors; i++) {
			if (Integer.valueOf(predictors[i]) == input.columnPredicted || Integer.valueOf(predictors[i]) >= input.datasetComplete.getDataMatrix().cols()) {
				continue;
			}
			input.columnPredictors[j++] = Integer.valueOf(predictors[i]);
		}

		if (j == 0) {
			throw new InvalidObjectException("There is no predictors!");
		}
//		System.out.println(j);
		input.columnPredictors = Arrays.copyOfRange(input.columnPredictors, 0, j);

//		for (int predictor : input.columnPredictors) {
//			System.out.println(predictor);
//		}
//		int k = 0;
//		for (int i = 0; i < datasetComplete.getDataMatrix().cols() - 1; i++) {
//			if (k != columnPredicted) {
//				columnPredictors[i] = k++;
//			} else {
//				columnPredictors[i] = ++k;
//				k++;
//			}
//		}
		return input;
	}

	private static void writeOutput (SimpleDataSet dataset) throws IOException {
		Scanner scanner = new Scanner(System.in);
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
