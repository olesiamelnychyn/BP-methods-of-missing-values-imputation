package com.company.imputationMethods;

import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class MultiDimensionalMultipleImputation extends ImputationMethod {
	int[] columnPredictors;
	SimpleDataSet datasetMissing;
	DataPoint dp;
	Statistics stat;
	ArrayList<ImputationMethod> methods = new ArrayList<>();


	public MultiDimensionalMultipleImputation (int columnPredicted, int[] columnPredictors, SimpleDataSet datasetMissing, DataPoint dp, Statistics stat, ArrayList<SimpleDataSet> datasets) {
		super(columnPredicted, datasets);
		this.columnPredicted = columnPredicted;
		this.datasetMissing = datasetMissing;
		this.columnPredictors = columnPredictors;
		this.dp = dp;
		this.stat = stat;
	}

	public void preprocessData () {
		// no preprocessing needed, it is done inside methods
	}

	public void fit () {
		SimpleImputationMethods simpleImputationMethods = new SimpleImputationMethods(datasetMissing);
		for (int columnPredictor : columnPredictors) {
			methods.add(simpleImputationMethods.imputeSimple(columnPredictor, dp, columnPredicted, stat, getDatasets()));
		}
	}

	public double predict (DataPoint dp) {
		Vec newValues = new DenseVector(columnPredictors.length);
		AtomicInteger i = new AtomicInteger();
		methods.forEach(method -> {
			method.preprocessData();
			method.fit();
			newValues.set(i.getAndIncrement(), method.predict(this.dp));
		});

		return newValues.mean();
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "MultiDimensional" + ANSI_RESET);
		System.out.println("Methods: " + methods.stream().map(ImputationMethod::getClass).map(Class::getName).collect(Collectors.joining(", ")));
	}
}