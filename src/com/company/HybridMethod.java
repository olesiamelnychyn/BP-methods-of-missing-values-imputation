package com.company;

import com.company.imputationMethods.*;
import com.company.utils.Evaluation;
import com.company.utils.calculations.StatCalculations;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static com.company.utils.ColorFormatPrint.ANSI_RED_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;
import static com.company.utils.calculations.MathCalculations.*;
import static com.company.utils.objects.PerformanceMeasures.df2;

/**
 * This class perform the main logic of the program - predict missing values
 */
public class HybridMethod {
	ConfigManager configManager = ConfigManager.getInstance();
	SimpleDataSet datasetMissing;
	int[] columnPredictors;
	Map<Integer, Statistics> statistics = new HashMap<>(); //Map of <column, statistics>
	boolean printOnlyFinal = Boolean.parseBoolean(configManager.get("input.printOnlyFinal"));
	Evaluation evaluation = null;
	MultipleImputationMethods multipleImputationMethods;
	SimpleImputationMethods simpleImputationMethods;

	public HybridMethod (int[] columnPredictors, SimpleDataSet datasetComplete, SimpleDataSet datasetMissing) {
		if (datasetComplete != null) {
			evaluation = new Evaluation(datasetComplete, printOnlyFinal);
		}
		this.datasetMissing = datasetMissing;
		this.columnPredictors = columnPredictors;
	}

	public void runImputation (int columnPredicted) throws IOException {
		// check if column predicted is not predicted
		if (columnPredicted != -1) {
			for (int i : columnPredictors) {
				if (i == columnPredicted) {
					System.out.println("Predictor cannot be predicted -- exit");
					return;
				}
			}
		}
		//create Statistics of column/-s
		statistics = StatCalculations.calcStatistics(columnPredicted, columnPredictors, datasetMissing);

		simpleImputationMethods = new SimpleImputationMethods(datasetMissing);
		multipleImputationMethods = new MultipleImputationMethods(datasetMissing, columnPredictors, Boolean.parseBoolean(configManager.get("impute.weighted")));

		int skipped = 0;
		for (DataPoint dp : datasetMissing.getDataPoints()) { //traverse dataset one by one

			if (columnPredicted != -1) { //if column to be imputed is specified
				if (Double.isNaN(dp.getNumericalValues().get(columnPredicted))) {
					impute(dp, columnPredicted);
				}
			} else { //all columns should be imputed
				int[] indexes = getIndexesOfNull(dp);
				if (getIntersection(indexes, columnPredictors).length != 0) {
					skipped++;
					if (!printOnlyFinal) {
						System.out.println(ANSI_RED_BACKGROUND + "Predictor cannot be predicted -- skip" + ANSI_RESET + "\n\n");
					}
				} else if (indexes.length > 0) {
					for (int idx : indexes) {
						impute(dp, idx);
					}
				}
			}
		}

		if (skipped != 0) {
			System.out.println("Number of records skipped due to absence of predictor : " + skipped);
		}

		if (evaluation != null) {
			evaluation.evaluateFinal(statistics);
		}
	}

	public void impute (DataPoint dp, int columnPredicted) {
		if (columnPredictors.length > 1) { //if it is multiple regression
			predict(multipleImputationMethods.imputeMultiple(dp, statistics.get(columnPredicted), columnPredicted));
		} else { //if it is simple regression (only one predictor)
			predict(simpleImputationMethods.imputeSimple(columnPredictors[0], dp, columnPredicted, statistics.get(columnPredicted)));
		}
	}

	protected void predict (ImputationMethod imputationMethod) {
		imputationMethod.preprocessData();
		imputationMethod.fit();
		if (!printOnlyFinal) {
			imputationMethod.print();
		}

		int columnPredicted = imputationMethod.getColumnPredicted();
		SimpleDataSet dataPoints = imputationMethod.getToBePredicted();
		for (DataPoint dp : dataPoints.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			double newValue = imputationMethod.predict(dp);

			if (Double.isNaN(newValue) && !(imputationMethod instanceof MeanImputationMethod)) { // returned value is NaN, try default method - mean imputation
				ArrayList<SimpleDataSet> dataSets = new ArrayList<>();
				dataSets.add(imputationMethod.getTrainingCopy());
				dataSets.add(dataPoints);
				predict(new MeanImputationMethod(columnPredicted, dataSets));
				continue;
			}

			dp.getNumericalValues().set(columnPredicted, getFormattedValue(newValue));
			if (evaluation != null) {
				evaluation.evaluate_concat(columnPredicted, indexMissing, dp);
			}
		}

		if (!printOnlyFinal) {
			System.out.println("\n");
		}
	}

	private double getFormattedValue (double value) {
		if (Double.isNaN(value)) {
			return value;
		}
		return Double.parseDouble(df2.format(value).replace(',', '.'));
	}
}
