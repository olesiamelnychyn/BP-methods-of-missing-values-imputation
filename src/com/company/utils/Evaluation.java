package com.company.utils;

import com.company.utils.objects.ImputedValue;
import com.company.utils.objects.PerformanceMeasures;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import static com.company.utils.ColorFormatPrint.*;
import static com.company.utils.objects.PerformanceMeasures.meanAbsolutePercentageError;

public class Evaluation {
	private Map<Integer, List<ImputedValue>> values = new HashMap<>(); //Map of <column, imputedValue>
	private final SimpleDataSet datasetComplete;
	private final SimpleDataSet datasetMissing;
	boolean printOnlyFinal;

	public Evaluation (SimpleDataSet datasetComplete, SimpleDataSet datasetMissing, boolean printOnlyFinal) {
		this.datasetComplete = datasetComplete;
		this.datasetMissing = datasetMissing;
		this.printOnlyFinal = printOnlyFinal;
	}

	/** Store predicted value in the evaluation object for calculating performance measures in the future
	 *
	 * @param columnPredicted
	 * @param toBePredicted
	 */
	public void evaluate_concat (int columnPredicted, DataPoint toBePredicted, double predictedValue) {
		int indexMissing = datasetMissing.getDataPoints().indexOf(toBePredicted);

		//if there is no complete dataset, there won't be any evaluation
		if (datasetComplete != null) {
			// otherwise, put it into list of imputed values of the appropriate column
			ImputedValue value = new ImputedValue(indexMissing, datasetComplete.getDataPoint(indexMissing).getNumericalValues().get(columnPredicted), predictedValue);
			if (values.containsKey(columnPredicted)) {
				values.get(columnPredicted).add(value);
			} else {
				List<ImputedValue> list = new ArrayList<>();
				list.add(value);
				values.put(columnPredicted, list);
			}
			if (!printOnlyFinal) {
				Vec test = new DenseVector(new double[]{value.actual});
				Vec predicted = new DenseVector(new double[]{value.predicted});
				System.out.println(value);
				System.out.println("\t" + indexMissing + "\t" + columnPredicted + "\t" + ANSI_BOLD_ON + ANSI_PURPLE + "Mean Absolute Percentage Error: " + format(meanAbsolutePercentageError(test, predicted)) + "%" + ANSI_RESET + ANSI_BOLD_OFF);
			}
		}
		toBePredicted.getNumericalValues().set(columnPredicted, predictedValue);
	}

	/**Final evaluation
	 * @throws IOException
	 *
	 * Performs evaluation of all predicted values
	 */
	public void evaluateFinal (Map<Integer, Statistics> statistics) throws IOException {
		System.out.println("\n" + ANSI_PURPLE_BACKGROUND + "Results" + ANSI_RESET + "\n");
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt"));
		writer.write("Results:\n");
		writer.close();

		// separate evaluation by columns
		for (Map.Entry<Integer, List<ImputedValue>> entry : values.entrySet()) {
			List<ImputedValue> list = entry.getValue();
			int columnPredicted = entry.getKey();
			Vec act = new DenseVector(list.size());
			Vec pred = new DenseVector(list.size());
			int i = 0;
			for (ImputedValue value : list) {
				if (!printOnlyFinal) {
					System.out.println(value);
				}
				act.set(i, value.actual);
				pred.set(i++, value.predicted);
			}
			PerformanceMeasures performanceMeasures = new PerformanceMeasures(act, pred, datasetComplete.getDataMatrix().getColumn(columnPredicted).mean(), columnPredicted);
			performanceMeasures.printAndWriteResults();
			System.out.println(statistics.get(columnPredicted));
		}
	}
}
