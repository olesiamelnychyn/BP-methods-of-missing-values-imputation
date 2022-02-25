package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.ArrayList;

import static com.company.utils.calculations.StatCalculations.hasLinearRelationship;

public class MultipleImputationMethods {
	private int[] columnPredictors;
	private SimpleDataSet datasetMissing;
	private boolean weighted;
	private ArrayList<SimpleDataSet> datasets;

	public MultipleImputationMethods (SimpleDataSet datasetMissing, int[] columnPredictors, boolean weighted) {
		this.columnPredictors = columnPredictors;
		this.datasetMissing = datasetMissing;
		this.weighted = weighted;
	}

	public ImputationMethod imputeMultiple (DataPoint dp, Statistics stat, int columnPredicted) {
		datasets = DatasetManipulation.getToBeImputedAndTrainDeepCopiesByClosestDistance(datasetMissing, datasetMissing.getDataPoints().indexOf(dp), columnPredicted, columnPredictors, 10, weighted);

		if (hasLinearRelationship(datasets.get(0), stat)) {
			return new MultipleLinearRegressionJSATMethod(columnPredicted, datasets, columnPredictors);
		} else {
			return new MultiDimensionalMultipleImputation(columnPredicted, columnPredictors, datasetMissing, dp, stat, datasets);
//			return new MultipleLinearRegressionJSATMethod(columnPredicted, datasets, columnPredictors, 2);
		}
	}
}
