package com.company;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

public class ConfigManager {
	private static ConfigManager single_instance = null;
	private static String filename = "src/com/company/config.properties";
	private Properties prop;

	private ConfigManager () {
		prop = new Properties();
		try {
			FileInputStream ip = new FileInputStream(filename);
			prop.load(ip);
		} catch (IOException e) {
			System.out.println("Check your configuration file");
			System.exit(-1);
		}
	}

	public String get (String key) {
		return prop.getProperty(key);
	}

	public void set (String key, String value) {
		prop.setProperty(key, value);
	}

	public void store () {
		try {
			FileOutputStream outputStream = new FileOutputStream(filename);
			prop.store(outputStream, null);
		} catch (IOException e) {
			System.out.println("The configuration was not saved");
		}
	}

	public static ConfigManager getInstance () {
		if (single_instance == null)
			single_instance = new ConfigManager();

		return single_instance;
	}

}
