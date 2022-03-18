package com.company.imputationMethods;

import com.company.utils.objects.MainData;
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
	MainData data;
	int[] columnPredictors;
	SimpleDataSet datasetMissing;
	DataPoint dp;
	Statistics stat;
	ArrayList<ImputationMethod> methods = new ArrayList<>();


	public MultiDimensionalMultipleImputation (MainData data, SimpleDataSet datasetMissing, Statistics stat) {
		super(data);
		this.data = data;
		this.datasetMissing = datasetMissing;
		this.stat = stat;
	}

	public void preprocessData () {
		// no preprocessing needed, it is done inside methods
	}

	public void fit () {
		// select best methods for each predictor separately
		SimpleImputationMethods simpleImputationMethods = new SimpleImputationMethods(datasetMissing);
		for (int columnPredictor : columnPredictors) {
			MainData mainData = new MainData(new int[]{columnPredictor}, columnPredicted, data.getDp(), data.getTrain(), data.getImpute());
			methods.add(simpleImputationMethods.imputeSimple(mainData, stat));
		}
	}

	public double predict (DataPoint dp) {
		Vec newValues = new DenseVector(columnPredictors.length);
		AtomicInteger i = new AtomicInteger();
		// get all the values predicted by the methods
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