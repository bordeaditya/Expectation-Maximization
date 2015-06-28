import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;


// Class for Clusters
class Clusters
{
	double N,alpha,mean,variance;

	public double getN() {
		return N;
	}

	public void setN(double n) {
		N = n;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public double getMean() {
		return mean;
	}

	public void setMean(double mean) {
		this.mean = mean;
	}

	public double getVariance() {
		return variance;
	}

	public void setVariance(double variance) {
		this.variance = variance;
	}
	
}

// Class for Points
class Points
{
	static int size;
	double value;
	double[] clusterWeight = new double[size];
	double[] probValue = new double[size];
	public double getValue() {
		return value;
	}
	
	public void setValue(double value) {
		this.value = value;
	}
	
}
public class EM {

	public static ArrayList<Points> dataPoints = new ArrayList<Points>();
	public static Clusters[] clusters;
	public static int k,h,i1;
	public static double likelihood =0,currentLikelihood =0;
	public static void main(String[] args) throws NumberFormatException, IOException {
		
		String path = args[0];
		k = Integer.parseInt(args[1]);
		Points.size = k;
		//**Set this value h=1 or h=2.
		h = 1; // Heuristic Type
		
		// fill the data points
		FillDataPoints(path);
		
		clusters = new Clusters[k];
		
		// Create K clusters
		for(int i=0;i<k;i++)
			clusters[i] = new Clusters();
		
		// Heuristic : 1
		// For General GMM
		int i=0;
		
		System.out.println("\n---- General GMM - Parameters : ----");
		InitializeAndApplyEM(false,1);

		System.out.println("\nFinal Parameters : ");
		for(Clusters c:clusters)
		{
			i++;
			System.out.println("*** Cluster : "+i+" ***");
			System.out.println("Mean : "+c.mean);
			System.out.println("Variance : "+c.variance);
		}
		System.out.println("Log-Likelihood = "+currentLikelihood);
		System.out.println("\tIterations = "+i1);
		
		ClearClusterPointData();
		// For Known Variance GMM
		System.out.println("\n\n---- Known Variance GMM - Parameters: ----");
		InitializeAndApplyEM(true,1);
		
		i=0;
		System.out.println("\nFinal Parameters : ");
		for(Clusters c:clusters)
		{
			i++;
			System.out.println("*** Cluster : "+i+" ***");
			System.out.println("Mean : "+c.mean);
			System.out.println("Variance : "+c.variance);
		}
		System.out.println("Log-Likelihood = "+currentLikelihood);
		System.out.println("\tIterations = "+i1);
		
	}

	// Clear Cluster and Point data
	private static void ClearClusterPointData() {
		
		for(int i=0;i<k;i++)
		{
			for(Points p:dataPoints)
			{
				p.clusterWeight[i]=0;
				p.probValue[i]=0;
			}
			clusters[i].setMean(0);
			clusters[i].setN(0);
			clusters[i].setAlpha(0);
			clusters[i].setVariance(0);
		}
		
	}

	private static void InitializeAndApplyEM(boolean isKnown, int h) {
		
		if(h==1) // Heuristic 1
		{
			// General GMM
			SelectKRandomPoints();
		
			SelectInitialAlpha();
		
			CalculateInitialvariance(isKnown);
		}
		else if(h==2) // Randomly Setting Wik // Heuristic 2
		{
			SetRandomWeights();
			CalculateMaximization(isKnown);
		}
		int i=0;
		System.out.println("\nInitial Parameters : ");
		for(Clusters c:clusters)
		{
			i++;
			System.out.println("*** Cluster : "+i+" ***");
			System.out.println("Mean : "+c.mean);
			System.out.println("Variance : "+c.variance);
		}
		i1=0;	
		/// EM- Iterative Steps
		do
		{
			i1++;
			likelihood = currentLikelihood;	
			// E - Step
			CalculateExpectaction();
			
			// M - Step
			CalculateMaximization(isKnown);
			
			// Log Likelihood
			currentLikelihood = CalculateLogLikelihood();
		 
		}while(Double.compare(currentLikelihood,likelihood)!=0);
		
	}

	// Setting Random Weights
	private static void SetRandomWeights() {
		
		for(Points p:dataPoints)
		{
			double normalizeFactor = 0;
			for(int i=0;i<k;i++)
			{
				p.clusterWeight[i] = GenerateRandom(0, 1);
				normalizeFactor = normalizeFactor + p.clusterWeight[i];
			}
			// Normalize Alpha value to make Summation Alpha = 1
			for(int i=0;i<k;i++)
				p.clusterWeight[i] = p.clusterWeight[i]/normalizeFactor;
		}
	}

	// Calculating Log Likelihood
	private static double CalculateLogLikelihood() {
		
		double logSum =0;
		for(Points p:dataPoints)
		{
			double sumVal =0;
			for(int i=0;i<k;i++)
				sumVal = sumVal + clusters[i].getAlpha() * p.probValue[i];
			
			logSum = logSum + Math.log(sumVal);
		}
		
		return logSum;
	}

	// Maximization
	private static void CalculateMaximization(boolean isKnownVar) {
		
		for(int i=0;i<k;i++)
		{
			// N
			clusters[i].setN(CalculateWeightSum(i,false,false));
			
			// Alpha
			clusters[i].setAlpha(clusters[i].getN()/dataPoints.size());
			
			// Mean
			clusters[i].setMean(CalculateWeightSum(i,true,false));
			
			// Variance
			if(!isKnownVar)
				clusters[i].setVariance(CalculateWeightSum(i,true,true));
			else // Known Variance Case
				clusters[i].setVariance(1.0);
		}
	}

	private static double CalculateWeightSum(int clusterNum, boolean isMult, boolean isVar) {
		
		double result =0;
		if(!isMult)
		{
			for(Points p:dataPoints)
				result = result + p.clusterWeight[clusterNum];
		}
		else // Multiplication of Value -  true
		{
			if(!isVar)
			{
				for(Points p:dataPoints)
					result = result + p.clusterWeight[clusterNum] * p.getValue();
				
			}
			else // If Variance
			{
				for(Points p:dataPoints)
					result = result + p.clusterWeight[clusterNum] * Math.pow(p.getValue()-clusters[clusterNum].getMean(),2);
			}
			if(Double.compare(clusters[clusterNum].getN(),0)!=0)
				result = result / clusters[clusterNum].getN();
			else
				result = 0;
		}
		return result;
	}

	// Expectation
	private static void CalculateExpectaction() {
		
		// Calculate Probability Value for Every Point
		for(Points p:dataPoints)
			CalculateClusterPointProbability(p);
		
		for(Points p:dataPoints)
			CalculateClusterPointWeight(p);
		
	}

	// Each Point - Cluster  wise Weight
	private static void CalculateClusterPointWeight(Points p) {
		
		double totalProb =0;
		for(int i=0;i<k;i++)
		{
			totalProb = totalProb + p.probValue[i] * clusters[i].getAlpha();
		}
		
		for(int i=0;i<k;i++)
		{
			double nR = clusters[i].getAlpha() * p.probValue[i];
			if(Double.compare(totalProb, 0)!=0)
				p.clusterWeight[i] = nR / totalProb;
			else
				p.clusterWeight[i] =0;
		}
	}

	// Calculating probabilities
	private static void CalculateClusterPointProbability(Points p) {
		
		for(int i=0;i<k;i++)
		{
			double nR = -1 * Math.pow(p.getValue()-clusters[i].getMean(),2);
			double dR = 2 * clusters[i].getVariance();
			double totalnR = Math.pow(Math.E, (nR/dR));
			double totaldR = Math.sqrt(2 * Math.PI) * Math.sqrt(Math.abs(clusters[i].getVariance()));
			if(Double.compare(totaldR, 0)!=0)
				p.probValue[i] = totalnR / totaldR;
			else
				p.probValue[i] = 0;
		}
	}

	// Calculate initial Variance
	private static void CalculateInitialvariance(boolean isKnown) {
		
		if(!isKnown)
		{
			for(int i=0;i<k;i++)
			{
				double variance = 0;
				for(Points p:dataPoints)
				{
					variance = variance + Math.pow(p.getValue() - clusters[i].getMean(), 2);
				}
				variance = variance / (dataPoints.size());
				clusters[i].setVariance(variance);
			}
		}
		else // variance is Known = 1 
		{
			for(int i=0;i<k;i++)
				clusters[i].setVariance(1.0);
		}
	}

	// Select initial Alpha Values
	private static void SelectInitialAlpha() {
		
		double normalizeFactor = 0;
		for(int i=0;i<k;i++)
		{
			clusters[i].setAlpha(GenerateRandom(0, 1));
			normalizeFactor = normalizeFactor + clusters[i].getAlpha();
		}
		// Normalize Alpha value to make Summation Alpha = 1
		for(int i=0;i<k;i++)
			clusters[i].setAlpha(clusters[i].getAlpha()/normalizeFactor);
		
		//for(int i=0;i<k;i++)
		//	clusters[i].setAlpha(1.0/k);

	}

	// Select K Random points
	private static void SelectKRandomPoints() {
		
		ArrayList<Integer> randomPoints = new ArrayList<Integer>();
		int i;
		// Select Random Value for the cluster Mean
		while(randomPoints.size()!=k)
		{
			int randomValue = GetRandom(0, dataPoints.size());
			if(!randomPoints.contains(randomValue))
				randomPoints.add(randomValue);
		}
		
		//randomPoints.add(4);
		//randomPoints.add(10);
		//randomPoints.add(12);
		
		// Assign Mean to each cluster
		for(i=0;i<k;i++)
		{
			Points p = dataPoints.get(randomPoints.get(i));
			clusters[i].setMean(p.getValue());
			//clusters[i].setMean(Math.random());
		}
	}

	// Generate Random integer
	private static int GetRandom(int min, int max) {
		 Random random = new Random();
		 int randomNum = random.nextInt((max - min) + 1) + min;
		 return randomNum;
	}

	// Fill the Data points List
	private static void FillDataPoints(String path) throws NumberFormatException, IOException {
		BufferedReader br = new BufferedReader(new FileReader(path));
		String line;
		// Fill the List
		while((line = br.readLine())!= null)
		{
			// Remove Special Characters: Before adding the word
			double value = Double.parseDouble(line.trim());
			Points point = new Points();
			point.setValue(value);
			dataPoints.add(point);
		}
		br.close();
		
	}
	
	// Generate Random Numbers
	public static double GenerateRandom(int min,int max)
	{
		double defaultNum = 0.5;
		Random rNo = new Random();
		double randomNumber = min + (max - (min)) * rNo.nextDouble();
		if(Double.compare(randomNumber, 0)!=0)
			return randomNumber;
		else
			return defaultNum;
	}

}
