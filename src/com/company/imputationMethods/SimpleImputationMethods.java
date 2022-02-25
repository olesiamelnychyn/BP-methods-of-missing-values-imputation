package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.ArrayList;

import static com.company.utils.calculations.StatCalculations.*;

public class SimpleImputationMethods {
	private ArrayList<SimpleDataSet> datasets;
	SimpleDataSet datasetMissing;

	public SimpleImputationMethods (SimpleDataSet datasetMissing) {
		this.datasetMissing = datasetMissing;
	}

	public ImputationMethod imputeSimple (int columnPredictor, DataPoint dp, int columnPredicted, Statistics stat) {
		datasets = DatasetManipulation.getToBeImputedAndTrainDeepCopiesAroundIndex(datasetMissing, datasetMissing.getDataPoints().indexOf(dp), columnPredicted, new int[]{columnPredictor}, 8);

		if (isCloseToMean(datasets.get(0), stat)) {
			return new MeanImputationMethod(columnPredicted, datasets);
		} else if (isCloseToMedian(datasets.get(0), stat)) {
			return new MedianImputationMethod(columnPredicted, datasets);
		} else if (isStrictlyIncreasing(datasets.get(0), columnPredicted) && isStrictlyIncreasing(datasets.get(0), columnPredictor)) {
			return new LinearInterpolatorApacheMethod(columnPredicted, datasets, columnPredictor, true);
		} else if (isStrictlyDecreasing(datasets.get(0), columnPredicted) && isStrictlyDecreasing(datasets.get(0), columnPredictor)) {
			return new LinearInterpolatorApacheMethod(columnPredicted, datasets, columnPredictor, false);
		} else if (hasLinearRelationship(datasets.get(0), stat)) {
			return new LinearRegressionJSATMethod(columnPredicted, datasets, columnPredictor);
		} else {
			int order = getPolynomialOrder(datasets.get(0), stat);
			if (order != -1) {
				return new PolynomialCurveFitterApacheMethod(columnPredicted, datasets, columnPredictor, order);
			}
		}
		return new MeanImputationMethod(columnPredicted, datasets);
	}
}
