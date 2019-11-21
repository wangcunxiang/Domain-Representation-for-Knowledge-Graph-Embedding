#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>

using namespace std;

const float pi = 3.141592653589793238462643383;

void* train_TransE(void *con);

int bernFlag = 0;
int dimension=100;
int dimensionR=10;
string inPath="";
string outPath="";
int threads=1;
int traintimes=1000;
int traintimes_E=1000;
int traintimes_S=1000;
int traintimes_R=1000;
int nbatches=100;
float alpha=0.0001;
float rate=0.00001;
float margin_E=1;
float margin_S=1;
float margin_R=1;
bool initwithTransE=false;
bool outBinaryFlag=false;
bool isSTransE=0;
bool isTransE=0;
bool isTransR=0;
int isTransX=0;
bool withEllipsoid=0;
string note="";
string note1="";

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
};

Triple *trainHead, *traintail, *trainList;

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};


unsigned long long *next_random;

unsigned long long randd(int id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

float rand(float min, float max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float randn(float miu,float sigma, float min ,float max) {
	float x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

void norm(float * con) {
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}


void norm(float *con, int dimension) {
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

void norm(float *con, float *matrix) {
	float tmp, x;
	int last;
	x = 0;
	last = 0;
	for (int ii = 0; ii < dimensionR; ii++) {
		tmp = 0;
		for (int jj=0; jj < dimension; jj++) {
			tmp += matrix[last] * con[jj];
			last++;
		}
		x += tmp * tmp;
	}
	if (x>1) {
		float lambda = 1;
		for (int ii = 0, last = 0; ii < dimensionR; ii++, last += dimension) {
			tmp = 0;
			for (int jj = 0; jj < dimension; jj++)
				tmp += ((matrix[last + jj] * con[jj]) * 2);
			for (int jj = 0; jj < dimension; jj++) {
				matrix[last + jj] -= alpha * lambda * tmp * con[jj];
				con[jj] -= alpha * lambda * tmp * matrix[last + jj];
			}
		}
	}
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}


int relationTotal, entityTotal, tripleTotal;
int *freqRel, *freqEnt;
float *left_mean, *right_mean;
float *relationVec, *entityVec, *matrix_h,*matrix, *matrix_t;
float *relationVecDao, *entityVecDao, *matrix_hDao, *matrix_tDao,*matrixDao;
float *tmpValue;

void norm_S(int h, int t, int r, int j) {
		norm(relationVecDao + dimensionR * r, dimensionR);
		norm(entityVecDao + dimension * h, dimension);
		norm(entityVecDao + dimension * t, dimension);
		norm(entityVecDao + dimension * j, dimension);
		norm(entityVecDao + dimension * h, matrix_hDao + dimension * dimensionR * r);
		norm(entityVecDao + dimension * t, matrix_tDao + dimension * dimensionR * r);
		norm(entityVecDao + dimension * j, matrix_hDao + dimension * dimensionR * r);
		norm(entityVecDao + dimension * j, matrix_tDao + dimension * dimensionR * r);
}

void norm_R(int h, int t, int r, int j) {
		norm(relationVecDao + dimensionR * r, dimensionR);
		norm(entityVecDao + dimension * h, dimension);
		norm(entityVecDao + dimension * t, dimension);
		norm(entityVecDao + dimension * j, dimension);
		norm(entityVecDao + dimension * h, matrixDao + dimension * dimensionR * r);
		norm(entityVecDao + dimension * t, matrixDao + dimension * dimensionR * r);
		norm(entityVecDao + dimension * j, matrixDao + dimension * dimensionR * r);
}

void init(){
	printf("initialization:");
	if(isTransE)
		printf("	TransE\n");
	if(isSTransE)
		printf("	StransE\n");
	FILE *fin;
	int tmp;
	cout<<inPath<<endl;
	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);
	printf("relationTotal=%d\n",relationTotal);


	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);
	if(isTransE){
		relationVec = (float *)calloc(relationTotal * dimension, sizeof(float));
		for (int i = 0; i < relationTotal; i++) {
			for (int ii=0; ii<dimension; ii++)
				relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		}
		entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
		for (int i = 0; i < entityTotal; i++) {
			for (int ii=0; ii<dimension; ii++)
				entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
			norm(entityVec+i*dimension);
		}
	}


	if(isTransR){
		relationVec = (float *)calloc(relationTotal * dimensionR * 2 + entityTotal * dimension * 2 + relationTotal * dimension * dimensionR * 2, sizeof(float));
		relationVecDao = relationVec + relationTotal * dimensionR;
		entityVec = relationVecDao + relationTotal * dimensionR;
		  

		entityVecDao = entityVec + entityTotal * dimension;
		matrix = entityVecDao + entityTotal * dimension;
		matrixDao = matrix + dimension * dimensionR * relationTotal;
		for (int i = 0; i < relationTotal; i++) {
			for (int ii=0; ii < dimensionR; ii++)
				relationVec[i * dimensionR + ii] = randn(0, 1.0 / dimensionR, -6 / sqrt(dimensionR), 6 / sqrt(dimensionR));
		}
		for (int i = 0; i < entityTotal; i++) {
			for (int ii=0; ii < dimension; ii++)
				entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
			norm(entityVec + i * dimension, dimension);
		}

		for (int i = 0; i < relationTotal; i++)
			for (int j = 0; j < dimensionR; j++)
				for (int k = 0; k < dimension; k++)
					matrix[i * dimension * dimensionR + j * dimension + k] =  randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	}

	if(isSTransE){
		relationVec = (float *)calloc(relationTotal * dimensionR * 2 + entityTotal * dimension * 2 + relationTotal * dimension * dimensionR * 4, sizeof(float));
		relationVecDao = relationVec + relationTotal * dimensionR;
		entityVec = relationVecDao + relationTotal * dimensionR;
		entityVecDao = entityVec + entityTotal * dimension;
		matrix_h = entityVecDao + entityTotal * dimension;
		matrix_hDao = matrix_h + dimension * dimensionR * relationTotal;
		matrix_t = matrix_hDao + dimension * dimensionR * relationTotal;
		matrix_tDao = matrix_t + dimension * dimensionR * relationTotal;
		freqRel = (int *)calloc(relationTotal + entityTotal, sizeof(int));
		freqEnt = freqRel + relationTotal;

		for (int i = 0; i < relationTotal; i++) {
			for (int ii=0; ii < dimensionR; ii++)
				relationVec[i * dimensionR + ii] = randn(0, 1.0 / dimensionR, -6 / sqrt(dimensionR), 6 / sqrt(dimensionR));
		}
		for (int i = 0; i < entityTotal; i++) {
			for (int ii=0; ii < dimension; ii++)
				entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
			norm(entityVec + i * dimension, dimension);
		}

		for (int i = 0; i < relationTotal; i++)
			for (int j = 0; j < dimensionR; j++)
				for (int k = 0; k < dimension; k++)
					matrix_h[i * dimension * dimensionR + j * dimension + k] =  randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));

		for (int i = 0; i < relationTotal; i++)
			for (int j = 0; j < dimensionR; j++)
				for (int k = 0; k < dimension; k++)
					matrix_t[i * dimension * dimensionR + j * dimension + k] =  randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	
	}
	freqRel = (int *)calloc(relationTotal + entityTotal, sizeof(int));
	freqEnt = freqRel + relationTotal;

	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));
	traintail = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	for (int i = 0; i < tripleTotal; i++) {
		tmp = fscanf(fin, "%d", &trainList[i].h);
		tmp = fscanf(fin, "%d", &trainList[i].t);
		tmp = fscanf(fin, "%d", &trainList[i].r);
		freqEnt[trainList[i].t]++;
		freqEnt[trainList[i].h]++;
		freqRel[trainList[i].r]++;
		trainHead[i] = trainList[i];
		traintail[i] = trainList[i];
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(traintail, traintail + tripleTotal, cmp_tail());

	lefHead = (int *)calloc(entityTotal, sizeof(int));
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(int)*entityTotal);
	memset(rigTail, -1, sizeof(int)*entityTotal);
	for (int i = 1; i < tripleTotal; i++) {
		if (traintail[i].t != traintail[i - 1].t) {
			rigTail[traintail[i - 1].t] = i - 1;
			lefTail[traintail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[traintail[tripleTotal - 1].t] = tripleTotal - 1;

	left_mean = (float *)calloc(relationTotal * 2, sizeof(float));
	right_mean = left_mean + relationTotal;
	for (int i = 0; i < entityTotal; i++) {
		for (int j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (int j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (traintail[j].r != traintail[j - 1].r)
				right_mean[traintail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[traintail[lefTail[i]].r] += 1.0;
	}

	for (int i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
	printf("initialization done\n");
}

int Len;
int Batch;
float res;

float calc_sum(int e1, int e2, int rel) {
	float sum=0;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastr = rel * dimension;
	for (int ii=0; ii < dimension; ii++)
		sum += fabs(entityVec[last2 + ii] - entityVec[last1 + ii] - relationVec[lastr + ii]);
	return sum;
}

float calc_sum(int e1, int e2, int rel, float *tmp1, float *tmp2) {
	int lastM = rel * dimension * dimensionR;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastr = rel * dimensionR;
	float sum = 0;
	for (int ii = 0; ii < dimensionR; ii++) {
		tmp1[ii] = tmp2[ii] = 0;
		for (int jj = 0; jj < dimension; jj++) {
			tmp1[ii] += matrix_h[lastM + jj] * entityVec[last1 + jj];
			tmp2[ii] += matrix_t[lastM + jj] * entityVec[last2 + jj];
		}
		lastM += dimension;
		sum += fabs(tmp1[ii] + relationVec[lastr + ii] - tmp2[ii]);
	}
	return sum;
}

void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimension;
	int lastb1 = e1_b * dimension;
	int lastb2 = e2_b * dimension;
	int lastbr = rel_b * dimension;
	for (int ii=0; ii  < dimension; ii++) {
		float x;
		x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);
		if (x > 0)
			x = -alpha;
		else
			x = alpha;
		relationVec[lastar + ii] -= x;
		entityVec[lasta1 + ii] -= x;
		entityVec[lasta2 + ii] += x;
		x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);
		if (x > 0)
			x = alpha;
		else
			x = -alpha;
		relationVec[lastbr + ii] -=  x;
		entityVec[lastb1 + ii] -= x;
		entityVec[lastb2 + ii] += x;
	}
}

void gradient(int e1_a, int e2_a, int rel_a, int belta, int same, float *tmp1, float *tmp2) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimensionR;
	int lastM = rel_a * dimensionR * dimension;
	float x;
	for (int ii=0; ii < dimensionR; ii++) {
		x = tmp2[ii] - tmp1[ii] - relationVec[lastar + ii];
		if (x > 0)
			x = belta * alpha;
		else
			x = -belta * alpha;
		for (int jj = 0; jj < dimension; jj++) {
			matrix_hDao[lastM + jj] -=  x * (entityVec[lasta1 + jj] );
			matrix_tDao[lastM + jj] -=  x * ( - entityVec[lasta2 + jj]);
			entityVecDao[lasta1 + jj] -= x * matrix_h[lastM + jj];
			entityVecDao[lasta2 + jj] += x * matrix_t[lastM + jj];
		}
		relationVecDao[lastar + ii] -= same * x;
		lastM = lastM + dimension;
	}
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
	float sum1 = calc_sum(e1_a, e2_a, rel_a);
	float sum2 = calc_sum(e1_b, e2_b, rel_b);
	if (sum1 + margin_E > sum2) {
		res += margin_E + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
	}
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, float *tmp) {
	float sum1 = calc_sum(e1_a, e2_a, rel_a, tmp, tmp + dimensionR);
	float sum2 = calc_sum(e1_b, e2_b, rel_b, tmp + dimensionR * 2, tmp + dimensionR * 3);
	if (sum1 + margin_S > sum2) {
		res += margin_S + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, -1, 1, tmp, tmp + dimensionR);
		gradient(e1_b, e2_b, rel_b, 1, 1, tmp + dimensionR * 2, tmp + dimensionR * 3);
	}
}


int transRLen;
int transRBatch;

float calc_sum_R(int e1, int e2, int rel, float *tmp1, float *tmp2) {
	int lastM = rel * dimension * dimensionR;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastr = rel * dimensionR;
	float sum = 0;
	for (int ii = 0; ii < dimensionR; ii++) {
		tmp1[ii] = tmp2[ii] = 0;
		for (int jj = 0; jj < dimension; jj++) {
			tmp1[ii] += matrix[lastM + jj] * entityVec[last1 + jj];
			tmp2[ii] += matrix[lastM + jj] * entityVec[last2 + jj];
		}
		lastM += dimension;
		sum += fabs(tmp1[ii] + relationVec[lastr + ii] - tmp2[ii]);
	}
	return sum;
}

void gradient_R(int e1_a, int e2_a, int rel_a, int belta, int same, float *tmp1, float *tmp2) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimensionR;
	int lastM = rel_a * dimensionR * dimension;
	float x;
	for (int ii=0; ii < dimensionR; ii++) {
		x = tmp2[ii] - tmp1[ii] - relationVec[lastar + ii];
		if (x > 0)
			x = belta * alpha;
		else
			x = -belta * alpha;
		for (int jj = 0; jj < dimension; jj++) {
			matrixDao[lastM + jj] -=  x * (entityVec[lasta1 + jj] - entityVec[lasta2 + jj]);
			entityVecDao[lasta1 + jj] -= x * matrix[lastM + jj];
			entityVecDao[lasta2 + jj] += x * matrix[lastM + jj];
		}
		relationVecDao[lastar + ii] -= same * x;
		lastM = lastM + dimension;
	}
}

void train_kb_R(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, float *tmp) {
	float sum1 = calc_sum_R(e1_a, e2_a, rel_a, tmp, tmp + dimensionR);
	float sum2 = calc_sum_R(e1_b, e2_b, rel_b, tmp + dimensionR * 2, tmp + dimensionR * 3);
	if (sum1 + margin_R > sum2) {
		res += margin_R + sum1 - sum2;
		gradient_R(e1_a, e2_a, rel_a, -1, 1, tmp, tmp + dimensionR);
    	gradient_R(e1_b, e2_b, rel_b, 1, 1, tmp + dimensionR * 2, tmp + dimensionR * 3);
	}
}

int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (traintail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (traintail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < traintail[ll].h) return tmp;
	if (tmp > traintail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (traintail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* trainMode_StranE(void *con) {
	int id, i, j, pr;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	float *tmp = tmpValue + id * dimensionR * 4;
	for (int k = Batch / threads; k >= 0; k--) {
		i = rand_max(id, Len);
		if (bernFlag)
			pr = 1000*right_mean[trainList[i].r]/(right_mean[trainList[i].r]+left_mean[trainList[i].r]);
		else
			pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r, tmp);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r, tmp);
		}
		norm_S(trainList[i].h, trainList[i].t, trainList[i].r, j);
	}
	pthread_exit(NULL);
}

void* trainMode_TransE(void *con) {
	int id, pr, i, j;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	for (int k = Batch / threads; k >= 0; k--) {
		i = rand_max(id, Len);
		if (bernFlag)
			pr = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
		else
			pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r);
		}
		norm(relationVec + dimension * trainList[i].r);
		norm(entityVec + dimension * trainList[i].h);
		norm(entityVec + dimension * trainList[i].t);
		norm(entityVec + dimension * j);
	}
	pthread_exit(NULL);
}

void* train_StranE(void *con) {
	printf("train paras StransE\n");
	if(initwithTransE){
		printf("initwithTransE\n");
		//train_TransE(NULL);
		for (int i = 0; i < relationTotal; i++)
			for (int j = 0; j < dimensionR; j++)
				for (int k = 0; k < dimension; k++)
					if (j == k)
					{
						matrix_h[i * dimension * dimensionR + j * dimension + k] = 1;
						matrix_t[i * dimension * dimensionR + j * dimension + k] = 1;
					}
					else
					{
						matrix_h[i * dimension * dimensionR + j * dimension + k] = 0;
						matrix_t[i * dimension * dimensionR + j * dimension + k] = 0;
					}
	}
	Len = tripleTotal;
	Batch = Len / nbatches;
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));
	tmpValue = (float *)calloc(threads * dimensionR * 4, sizeof(float));
	memcpy(relationVecDao, relationVec, dimensionR * relationTotal * sizeof(float));
	memcpy(entityVecDao, entityVec, dimension * entityTotal * sizeof(float));
	memcpy(matrix_hDao, matrix_h, dimension * relationTotal * dimensionR * sizeof(float));
	memcpy(matrix_tDao, matrix_t, dimension * relationTotal * dimensionR * sizeof(float));
	for (int epoch = 1; epoch <=traintimes_S; epoch++) {
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
			for (long a = 0; a < threads; a++)
				pthread_create(&pt[a], NULL, trainMode_StranE,  (void*)a);
			for (long a = 0; a < threads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			memcpy(relationVec, relationVecDao, dimensionR * relationTotal * sizeof(float));
			memcpy(entityVec, entityVecDao, dimension * entityTotal * sizeof(float));
			memcpy(matrix_h, matrix_hDao, dimension * relationTotal * dimensionR * sizeof(float));
			memcpy(matrix_t, matrix_tDao, dimension * relationTotal * dimensionR * sizeof(float));
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

void* train_TransE(void *con) {
	if(isTransE)
		cout<<"train paras of TransE\n";
	else
		cout<<"init for St ransE\n";
	Len = tripleTotal;
	Batch = Len / nbatches;int iii=573;
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));//cout<<++iii<<endl;
	for (int epoch = 0; epoch < traintimes_E; epoch++) {
		res = 0;//cout<<++iii<<endl;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
		//	cout<<"threads:"<<threads<<endl;
			for (long a = 0; a < threads; a++){
		//		cout<<"a:"<<a<<endl;
				pthread_create(&pt[a], NULL, trainMode_TransE, (void*)a);
			}
		//	cout<<"dkkkk\n";
			for (long a = 0; a < threads; a++){
		//		cout<<"a1:"<<a<<endl;
				pthread_join(pt[a], NULL);
		//		cout<<"xxx\n";
			}
			free(pt);
		}
		printf("epoch %d %f\n", epoch, res);//cout<<++iii<<endl;
	}
}


void* trainMode_R(void *con) {
	int id, i, j, pr;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	float *tmp = tmpValue + id * dimensionR * 4;
	for (int k = transRBatch / threads; k >= 0; k--) {
		i = rand_max(id, transRLen);	
		if (bernFlag)
			pr = 1000*right_mean[trainList[i].r]/(right_mean[trainList[i].r]+left_mean[trainList[i].r]);
		else
			pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb_R(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r, tmp);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb_R(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r, tmp);
		}
		norm_R(trainList[i].h, trainList[i].t, trainList[i].r, j);
	}
	pthread_exit(NULL);
}

void* train_R(void *con) {
	transRLen = tripleTotal;
	transRBatch = transRLen / nbatches;
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));
	tmpValue = (float *)calloc(threads * dimensionR * 4, sizeof(float));
	memcpy(relationVecDao, relationVec, dimensionR * relationTotal * sizeof(float));
	memcpy(entityVecDao, entityVec, dimension * entityTotal * sizeof(float));
	memcpy(matrixDao, matrix, dimension * relationTotal * dimensionR * sizeof(float));
	for (int epoch = 0; epoch < traintimes_R; epoch++) {
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
			for (long a = 0; a < threads; a++)
				pthread_create(&pt[a], NULL, trainMode_R,  (void*)a);
			for (long a = 0; a < threads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			memcpy(relationVec, relationVecDao, dimensionR * relationTotal * sizeof(float));
			memcpy(entityVec, entityVecDao, dimension * entityTotal * sizeof(float));
			memcpy(matrix, matrixDao, dimension * relationTotal * dimensionR * sizeof(float));
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

void* train(void *con){
	if(isTransE)
		return train_TransE(con);
	if(isSTransE)
		return train_StranE(con);
	if(isTransR)
		return train_R(con);
}

void out_binary_TransE() {
		int len, tot;
		float *head;
		FILE* f2 = fopen((outPath + "relation2vec" + note +note1+ ".bin").c_str(), "wb");
		FILE* f3 = fopen((outPath + "entity2vec" + note +note1+ ".bin").c_str(), "wb");
		len = relationTotal * dimension; tot = 0;
		head = relationVec;
		while (tot < len) {
			int sum = fwrite(head + tot, sizeof(float), len - tot, f2);
			tot = tot + sum;
		}
		len = entityTotal * dimension; tot = 0;
		head = entityVec;
		while (tot < len) {
			int sum = fwrite(head + tot, sizeof(float), len - tot, f3);
			tot = tot + sum;
		}
		fclose(f2);
		fclose(f3);
}

void out_TransE() {
		if (outBinaryFlag) {
			out_binary_TransE();
			return;
		}
		FILE* f2 = fopen((outPath + "relation2vec" + note+note1 + ".vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "entity2vec" + note+note1 + ".vec").c_str(), "w");
		for (int i=0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
}

void out_binary_StransE() {
		int len, tot;
		float *head;
		FILE* f2 = fopen((outPath + "relation2vec" + note+note1 + ".bin").c_str(), "wb");
		FILE* f3 = fopen((outPath + "entity2vec" + note +note1+ ".bin").c_str(), "wb");
		len = relationTotal * dimension; tot = 0;
		head = relationVec;
		while (tot < len) {
			int sum = fwrite(head + tot, sizeof(float), len - tot, f2);
			tot = tot + sum;
		}
		len = entityTotal * dimension; tot = 0;
		head = entityVec;
		while (tot < len) {
			int sum = fwrite(head + tot, sizeof(float), len - tot, f3);
			tot = tot + sum;
		}
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen((outPath + "A" + note +note1+ ".bin").c_str(), "wb");
		len = relationTotal * dimension * dimensionR; tot = 0;
		head = matrix_h;
		while (tot < len) {
			int sum = fwrite(head + tot, sizeof(float), len - tot, f1);
			tot = tot + sum;
		}
		fclose(f1);
}

void out_StransE() {
		if (outBinaryFlag) {
			out_binary_StransE();
			return;
		}
		FILE* f2 = fopen((outPath + "relation2vec" + note +note1+ ".vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "entity2vec" + note+note1 + ".vec").c_str(), "w");
		for (int i = 0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen((outPath + "A" + note+note1 + ".vec").c_str(),"w");
		for (int i = 0; i < relationTotal; i++)
			for (int jj = 0; jj < dimension; jj++) {
				for (int ii = 0; ii < dimensionR; ii++)
					fprintf(f1, "%f\t", matrix_h[i * dimensionR * dimension + jj + ii * dimension]);
				fprintf(f1,"\n");
			}
		fclose(f1);
		FILE* f4 = fopen((outPath + "B" + note+note1 + ".vec").c_str(),"w");
		for (int i = 0; i < relationTotal; i++)
			for (int jj = 0; jj < dimension; jj++) {
				for (int ii = 0; ii < dimensionR; ii++)
					fprintf(f4, "%f\t", matrix_t[i * dimensionR * dimension + jj + ii * dimension]);
				fprintf(f1,"\n");
			}
		fclose(f4);
}

void out_binary_TransR() {
		int len, tot;
		float *head;		
		FILE* f2 = fopen((outPath + "relation2vec" + note +note1+ ".bin").c_str(), "wb");
		FILE* f3 = fopen((outPath + "entity2vec" + note+note1 + ".bin").c_str(), "wb");
		len = relationTotal * dimension; tot = 0;
		head = relationVec;
		while (tot < len) {
			int sum = fwrite(head + tot, sizeof(float), len - tot, f2);
			tot = tot + sum;
		}
		len = entityTotal * dimension; tot = 0;
		head = entityVec;
		while (tot < len) {
			int sum = fwrite(head + tot, sizeof(float), len - tot, f3);
			tot = tot + sum;
		}	
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen((outPath + "A" + note +note1+ ".bin").c_str(), "wb");
		len = relationTotal * dimension * dimensionR; tot = 0;
		head = matrix;
		while (tot < len) {
			int sum = fwrite(head + tot, sizeof(float), len - tot, f1);
			tot = tot + sum;
		}
		fclose(f1);
}

void out_TransR() {
		if (outBinaryFlag) {
			out_binary_TransR(); 
			return;
		}


		FILE* f2 = fopen((outPath + "relation2vec" + note+note1 + ".vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "entity2vec" + note +note1+ ".vec").c_str(), "w");
		for (int i = 0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen((outPath + "A" + note +note1+ ".vec").c_str(),"w");
		for (int i = 0; i < relationTotal; i++)
			for (int jj = 0; jj < dimension; jj++) {
				for (int ii = 0; ii < dimensionR; ii++)
					fprintf(f1, "%f\t", matrix[i * dimensionR * dimension + jj + ii * dimension]);
				fprintf(f1,"\n");
			}
		fclose(f1);
}

void out(){
	if(isTransE)
		out_TransE();
	if(isSTransE)
		out_StransE();
	if(isTransR)
		out_TransR();
}

float *ellip_matrix_h,*ellip_matrix_t,*ellip_center_h,*ellip_center_t;
float *tmp_h,*tmp_t,*tmpm_h,*tmpm_t,*distance_h,*distance_t,*socre_h,*socre_t;


void* prepareMode_S(void *con) {
	long id;
	id = (unsigned long long)(con);
	long lef = entityTotal / (threads) * id;
	long rig = entityTotal / (threads) * (id + 1);
	if (id == threads - 1) rig = entityTotal;
	for (long i = lef; i < rig; i++) {
		for (long j = 0; j < relationTotal; j++) {
			long last = i * dimensionR * relationTotal + j * dimensionR;
			for (long k = 0; k < dimensionR; k++)
				for (long kk = 0; kk < dimension; kk++)
				{
					tmp_h[last + k] += matrix_h[j * dimension * dimensionR + k * dimension + kk] * entityVec[i * dimension + kk];
					tmp_t[last + k] += matrix_t[j * dimension * dimensionR + k * dimension + kk] * entityVec[i * dimension + kk];
				}

		}
	}
	pthread_exit(NULL);
}

void* prepareMode_R(void *con) {
    long id;
    id = (unsigned long long)(con);
    long lef = entityTotal / (threads) * id;
    long rig = entityTotal / (threads) * (id + 1);
    if (id == threads - 1) rig = entityTotal;
    for (long i = lef; i < rig; i++) {
        for (long j = 0; j < relationTotal; j++) {
            long last = i * dimensionR * relationTotal + j * dimensionR;
            for (long k = 0; k < dimensionR; k++)
                for (long kk = 0; kk < dimension; kk++)
                    tmp_h[last + k] += matrix[j * dimension * dimensionR + k * dimension + kk] * entityVec[i * dimension + kk];
        }
    }
    pthread_exit(NULL);
}

void pepare(){
	int num=dimension * (dimension + 1) / 2;
	cout<<"pepare ellipsoid\n";
		ellip_matrix_h=(float *)calloc(long(relationTotal)* num*2+relationTotal*dimension*2,sizeof(float));
		ellip_matrix_t=ellip_matrix_h+relationTotal* num;

		ellip_center_h=ellip_matrix_t+relationTotal*num;
		ellip_center_t=ellip_center_h+relationTotal*dimension;

	if(isSTransE){
		tmp_h = (float *)calloc(long(relationTotal)*entityTotal*dimension*2,sizeof(float));
		tmp_t = tmp_h+relationTotal*entityTotal*dimension;
		pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
		for (long a = 0; a < threads; a++)
			pthread_create(&pt[a], NULL, prepareMode_S,  (void*)a);
		for (long a = 0; a < threads; a++)
			pthread_join(pt[a], NULL);
		free(pt);
	}

	if(isTransR){
		tmp_h=(float *)calloc(long(relationTotal)*entityTotal*dimension,sizeof(float));
		tmp_t=tmp_h;
		pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
		for (long a = 0; a < threads; a++)
			pthread_create(&pt[a], NULL, prepareMode_R,  (void*)a);
		for (long a = 0; a < threads; a++)
			pthread_join(pt[a], NULL);
		free(pt);
	}

	tmpm_h=(float *)calloc(num*2*relationTotal,sizeof(float));
	tmpm_t=tmpm_h+num*relationTotal;

	for (int i = 0; i < 2 * relationTotal; i++){
		int last=i * num;
		for(int ii = 0;ii < dimension; ii++)
			for(int jj = ii; jj < dimension; jj++){
				if(ii==jj){
					ellip_matrix_h[last+ii*dimension+jj-ii*(ii+1)/2]=0.001+fabs(randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension)));
				}
				else
					ellip_matrix_h[last+ii*dimension+jj-ii*(ii+1)/2]=randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
			}
	}
	for (int i = 0; i < 2 * relationTotal; i++){
		for (int ii=0; ii < dimension; ii++){
			ellip_center_h[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		}
	}
	cout<<"pepare done\n";
}

void cal_M(int rel,float *M_h,float *M_t){
	int lastM = rel * dimension * (dimension+1)/2;
	for(int i=0;i<dimension;i++)
		for(int j=i;j<dimension;j++){
			for(int k=0;k<=i;k++){
				M_h[i * dimension + j -  i*(i+1)/2] += ellip_matrix_h[lastM + k*dimension+i-k*(k+1)/2]*ellip_matrix_h[lastM + k*dimension+j-k*(k+1)/2];
				if(isnan(M_h[i * dimension + j -  i*(i+1)/2])) cerr<<"error";
			}
		}
	for(int i=0;i<dimension;i++)
		for(int j=i;j<dimension;j++)
			for(int k=0;k<=i;k++){
				M_t[i * dimension + j -  i*(i+1)/2] += ellip_matrix_t[lastM + k*dimension+i-k*(k+1)/2]*ellip_matrix_t[lastM + k*dimension+j-k*(k+1)/2];
			}
}



float cal_k_h(int e,int rel, float *M){
	float *eVec;
	if(isSTransE||isTransR){
		eVec=tmp_h+rel*dimension+e*relationTotal*dimension;
	}
	if(isTransE){
		eVec=entityVec+e*dimension;
	}

	int lastc = rel * dimension;
	float k=0;
	for(int i=0;i<dimension;i++)
		for(int j=i;j<dimension;j++){
			k+=2*M[i * dimension + j -  i*(i+1)/2]*(eVec[i]-ellip_center_h[lastc+i])*(eVec[j]-ellip_center_h[lastc+j]);
		}
		
	return k;
}
float cal_k_t(int e,int rel, float *M){
	float *eVec;
	if(isSTransE||isTransR){
		eVec=tmp_t+rel*dimension+e*relationTotal*dimension;
	}
	if(isTransE){
		eVec=entityVec+e*dimension;
	}

	int lastc = rel * dimension;
	float k=0;
	for(int i=0;i<dimension;i++)
		for(int j=i;j<dimension;j++)
			k+=2*M[i * dimension + j -  i*(i+1)/2]*(eVec[i]-ellip_center_t[lastc+i])*(eVec[j]-ellip_center_t[lastc+j]);
	return k;
}

float cal_y_h(int e,int rel){
	float *eVec;
	if(isSTransE||isTransR){
		eVec=tmp_h+rel*dimension+e*relationTotal*dimension;
	}
	if(isTransE){
		eVec=entityVec+e*dimension;
	}

	int lastc =  rel *dimension;
	float y=0;
	for(int i=0;i<dimension;i++)
		y+=(eVec[i]-ellip_center_h[lastc+i])*(eVec[i]-ellip_center_h[lastc+i]);
	return y;
}
float cal_y_t(int e,int rel){
	float *eVec;
	if(isSTransE||isTransR){
		eVec=tmp_t+rel*dimension+e*relationTotal*dimension;
	}
	if(isTransE){
		eVec=entityVec+e*dimension;
	}

	int lastc =  rel *dimension;
	float y=0;
	for(int i=0;i<dimension;i++)
		y+=(eVec[i]-ellip_center_t[lastc+i])*(eVec[i]-ellip_center_t[lastc+i]);
	return y;
}

float getsocre(int eh, int et, int rel){

	float M_h[dimension*(dimension+1)/2];
	float M_t[dimension*(dimension+1)/2];

	memset(M_h,0,dimension*(dimension+1)/2*sizeof(float));
	memset(M_t,0,dimension*(dimension+ 1)/2*sizeof(float));

	cal_M(rel, M_h,M_t);



	float k_h=cal_k_h(eh, rel, M_h);
	float k_t=cal_k_t(et, rel,M_t);
	float y_h=cal_y_h(eh,rel);
	float y_t=cal_y_t(et,rel);

	float d_h=pow(1-pow(k_h,-0.5),2)*y_h;
	float d_t=pow(1-pow(k_t,-0.5),2)*y_t;

	return d_h+d_t;
}

void gradient_ellipsoid(int eh, int et, int rel) {

	int lastM = rel * dimension * (dimension+1)/2;
	int lastc = rel * dimension;

	float *enti_h,*enti_t;
	if(isSTransE||isTransR){
		enti_h=tmp_h+rel*dimension+eh*relationTotal*dimension;
		enti_t=tmp_t+rel*dimension+et*relationTotal*dimension;
	}
	if(isTransE){
		enti_h=entityVec+eh*dimension;
		enti_t=entityVec+et*dimension;
	}

	float M_h[dimension*(dimension+1)/2];
	float M_t[dimension*(dimension+1)/2];

	memset(M_h,0,dimension*(dimension+1)/2*sizeof(float));
	memset(M_t,0,dimension*(dimension+ 1)/2*sizeof(float));

	cal_M(rel, M_h,M_t);

	memcpy(tmpm_h+lastM,M_h,dimension*(dimension+1)/2*sizeof(float));
	memcpy(tmpm_t+lastM,M_t,dimension*(dimension+1)/2*sizeof(float));

	float k_h=cal_k_h(eh, rel, M_h);
	float k_t=cal_k_t(et, rel,M_t);
	float y_h=cal_y_h(eh,rel);
	float y_t=cal_y_t(et,rel);

	for(int i=0;i<dimension;i++)
		for(int j=i;j<dimension;j++)
		{	
			float z=0;
			for(int k=i;k<dimension;k++)
				z+=2*(enti_h[k]-ellip_center_h[lastc+k])*(enti_h[j]-ellip_center_h[lastc+j])*ellip_matrix_h[lastM+i*dimension+k-i*(i+1)/2];
			ellip_matrix_h[lastM+i*dimension+j-i*(i+1)/2] -= rate * y_h * (1-pow(k_h,-0.5)) *pow(k_h,-1.5)*z;
		}
	for(int i=0;i<dimension;i++)
		for(int j=i;j<dimension;j++)
		{	
			float z=0;
			for(int k=i;k<dimension;k++)
				z+=2*(enti_t[k]-ellip_center_t[lastc+k])*(enti_t[j]-ellip_center_t[lastc+j])*ellip_matrix_t[lastM+i*dimension+k-i*(i+1)/2];
			ellip_matrix_t[lastM+i*dimension+j-i*(i+1)/2] -= rate * y_t * (1-pow(k_t,-0.5)) *pow(k_t,-1.5)*z;
		}

	for(int i=0;i<dimension;i++)
	{
		float z=0;
		for(int k=0;k<dimension;k++){
			if(k<=i)
				z+=2*ellip_matrix_h[k*dimension+i-k*(k+1)/2]*(enti_h[k]-ellip_center_h[lastc+k]);
			else
				z+=2*ellip_matrix_h[i*dimension+k-i*(i+1)/2]*(enti_h[k]-ellip_center_h[lastc+k]);
		}
		ellip_center_h[lastc+i]-=rate*(pow(1-pow(k_h,-0.5),2)*2*(enti_h[i]-ellip_center_h[lastc+i])+y_h*(1-pow(k_h,-0.5)) *pow(k_h,-1.5)*z);
	}
	for(int i=0;i<dimension;i++)
	{
		float z=0;
		for(int k=0;k<dimension;k++){
			if(k<=i)
				z+=2*ellip_matrix_t[k*dimension+i-k*(k+1)/2]*(enti_t[k]-ellip_center_t[lastc+k]);
			else
				z+=2*ellip_matrix_t[i*dimension+k-i*(i+1)/2]*(enti_t[k]-ellip_center_t[lastc+k]);
		}
		ellip_center_t[lastc+i]-=rate*(pow(1-pow(k_t,-0.5),2)*2*(enti_t[i]-ellip_center_t[lastc+i])+y_t*(1-pow(k_t,-0.5)) *pow(k_t,-1.5)*z);
	}
}

float res_elli=0;

void train_bk_ellipsoid(int eh_a, int et_a, int rel_a){

	float sum1 = getsocre(eh_a, eh_a, rel_a);

	res_elli+=sum1;
	gradient_ellipsoid(eh_a,et_a,rel_a);
}

void* trainMode_ellisoid(void* con){
	int id, pr, i, j;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	for (int k = Batch / threads; k >= 0; k--) {
		i = rand_max(id, Len);
		train_bk_ellipsoid(trainList[i].h, trainList[i].t, trainList[i].r);
	}
	pthread_exit(NULL);
}
void* train_ellipsoid(void* con){
	cout<<"train ellipsoid\n";
	Len = tripleTotal;
	Batch = Len / nbatches;
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));

	for (int epoch = 0; epoch < traintimes; epoch++) {

		res_elli = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			//res = 0;
			pthread_t *pt = (pthread_t *)malloc(threads * sizeof(pthread_t));
			for (long a = 0; a < threads; a++)
				pthread_create(&pt[a], NULL, trainMode_ellisoid,  (void*)a);
			for (long a = 0; a < threads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}
		printf("epoch %d %f\n", epoch, res_elli);
	}
}

void savevalue() {


		if(isTransE)
			note1="_TransE";
		if(isSTransE)
			note1="_STransE";
		if(isTransR)
			note1="_TransR";
		printf("savevalue:M\n");

		FILE* f1 = fopen((outPath + "M" + note+note1 + ".vec").c_str(),"w");
		for (int i = 0; i < 2 * relationTotal; i++)
			for (int jj = 0; jj < dimension; jj++) {
				for (int ii = jj; ii < dimension; ii++)
					fprintf(f1, "%.6f\t", tmpm_h[i * dimension * (dimension+1)/2 + jj*dimension + ii -jj*(jj+1)/2]);
				fprintf(f1,"\n");
			}
		fclose(f1);
				printf("savevalue:C\n");

		FILE* f4 = fopen((outPath + "C" + note+note1 + ".vec").c_str(),"w");
		for (int i = 0; i < relationTotal * 2; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f4, "%.6f\t", ellip_center_h[last + ii]);
			fprintf(f4,"\n");
		}
		fclose(f4);
		printf("save done\n");
}

void setparameters(int argc, char **argv) {
	int i;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sizeR", argc, argv)) > 0) dimensionR = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) inPath = argv[i + 1];
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) outPath = argv[i + 1];
	if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) traintimes = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epochs_E", argc, argv)) > 0) traintimes_E = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epochs_S", argc, argv)) > 0) traintimes_S = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epochs_R", argc, argv)) > 0) traintimes_R = atoi(argv[i + 1]);	
	if ((i = ArgPos((char *)"-nbatches", argc, argv)) > 0) nbatches = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-rate", argc, argv)) > 0) rate = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin_S", argc, argv)) > 0) margin_S = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin_E", argc, argv)) > 0) margin_E = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin_R", argc, argv)) > 0) margin_R = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-initwithTransE", argc, argv)) > 0) initwithTransE = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-isTransX", argc, argv)) > 0) isTransX = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-withEllipsoid", argc, argv)) > 0) withEllipsoid = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
}

int main(int argc, char **argv) {
	setparameters(argc, argv);

	switch(isTransX){
		case 0: isTransE=1;
				break;
		case 1: isTransR=1;
				break;
		case 2: isSTransE=1;
				break;
	}

	init();
	if(initwithTransE){
		printf("initwithTransE\n");
		train_TransE(NULL);
		printf("initTransEdone\n");
	}
	train(NULL);
	if(withEllipsoid){
		pepare();
		train_ellipsoid(NULL);
		savevalue();
	}
	if (outPath != "") out();
	return 0;
}
