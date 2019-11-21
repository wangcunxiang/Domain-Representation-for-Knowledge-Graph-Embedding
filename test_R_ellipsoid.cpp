#include <iostream>
#include <cstring>
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#define INT long
#define REAL float
using namespace std;

long relationTotal;
long entityTotal;
long Threads = 8;
long dimensionR = 50;
long dimension = 50;
float testid = 0;
long domain = 1;

float *entityVec, *relationVec, *entityRelVec, *matrix,*matrix_h,*matrix_t,*centr_h,*centr_t,*hEntDom,*tEntDom;
long testTotal, tripleTotal, trainTotal, validTotal;

struct Triple {
    long h, r, t, label;
};

struct cmp_head {
    bool operator()(const Triple &a, const Triple &b) {
        return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
    }
};

Triple *testList, *tripleList;
string initPath = "";
string loadPath = "./";
string inPath = "";
string note = "";
int nntotal[5];
int head_lef[10000];
int head_rig[10000];
int tail_lef[10000];
int tail_rig[10000];
int head_type[1000000];
int tail_type[1000000];

REAL cal_hEntDom(long e,long rel);
REAL cal_tEntDom(long e,long rel);

void init() {
    FILE *fin;
    long tmp, h, r, t, label;

    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);
    relationVec = (float *)calloc(relationTotal * dimensionR, sizeof(float));

    fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);
    entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
    matrix = (float *)calloc(relationTotal * dimension * dimensionR, sizeof(float));

	matrix_h = (float *)calloc(relationTotal*dimension*(dimension+1),sizeof(float));
    matrix_t = matrix_h + relationTotal *dimension *(dimension+1)/2+dimension;
    centr_h = (float *)calloc(2*relationTotal * dimension, sizeof(float));
    centr_t = centr_h+relationTotal * dimension+dimension;
	
	hEntDom = (float *)calloc(relationTotal * entityTotal,sizeof(float));
	tEntDom = (float *)calloc(relationTotal * entityTotal,sizeof(float));

    FILE* f_kb1 = fopen((inPath + "test2id_all.txt").c_str(),"r");
    FILE* f_kb2 = fopen((inPath + "train2id.txt").c_str(),"r");
    FILE* f_kb3 = fopen((inPath + "valid2id.txt").c_str(),"r");
    tmp = fscanf(f_kb1, "%ld", &testTotal);
    tmp = fscanf(f_kb2, "%ld", &trainTotal);
    tmp = fscanf(f_kb3, "%ld", &validTotal);
    tripleTotal = testTotal + trainTotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    memset(nntotal, 0, sizeof(nntotal));
    for (long i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &label);
        tmp = fscanf(f_kb1, "%ld", &h);
        tmp = fscanf(f_kb1, "%ld", &t);
        tmp = fscanf(f_kb1, "%ld", &r);
        label++;
        nntotal[label]++;
        testList[i].label = label;
        testList[i].h = h;
        testList[i].t = t;
        testList[i].r = r;
        tripleList[i].h = h;
        tripleList[i].t = t;
        tripleList[i].r = r;
    }
    for (long i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &h);
        tmp = fscanf(f_kb2, "%ld", &t);
        tmp = fscanf(f_kb2, "%ld", &r);
        tripleList[i + testTotal].h = h;
        tripleList[i + testTotal].t = t;
        tripleList[i + testTotal].r = r;
    }

    for (long i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &h);
        tmp = fscanf(f_kb3, "%ld", &t);
        tmp = fscanf(f_kb3, "%ld", &r);
        tripleList[i + testTotal + trainTotal].h = h;
        tripleList[i + testTotal + trainTotal].t = t;
        tripleList[i + testTotal + trainTotal].r = r;
    }

    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    sort(tripleList, tripleList + tripleTotal, cmp_head());

    long total_lef = 0;
    long total_rig = 0;
    FILE* f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    tmp = fscanf(f_type, "%ld", &tmp);
    for (int i = 0; i < relationTotal; i++) {
        int rel, tot;
        tmp = fscanf(f_type, "%d%d", &rel, &tot);
        head_lef[rel] = total_lef;
        for (int j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%d", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%d%d", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (int j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%d", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
    cout<<"inin\n";
}


void* prepareMode(void *con) {
    long id;
    id = (unsigned long long)(con);
    long lef = entityTotal / (Threads) * id;
    long rig = entityTotal / (Threads) * (id + 1);
    if (id == Threads - 1) rig = entityTotal;
    for (long i = lef; i < rig; i++) {
        for (long j = 0; j < relationTotal; j++) {
            long last = i * dimensionR * relationTotal + j * dimensionR;
            for (long k = 0; k < dimensionR; k++)
                for (long kk = 0; kk < dimension; kk++)
                    entityRelVec[last + k] += matrix[j * dimension * dimensionR + k * dimension + kk] * entityVec[i * dimension + kk];
				
			if(domain){
				hEntDom[ i * relationTotal + j] = cal_hEntDom(i,j);
				tEntDom[ i * relationTotal + j] = cal_tEntDom(i,j);
			}
        }
    }
    pthread_exit(NULL);
}

void prepare() {
    FILE *fin;
    long tmp;
    fin = fopen((initPath + "entity2vec" + note + ".vec").c_str(), "r");
    for (long i = 0; i < entityTotal; i++) {
        long last = i * dimension;
        for (long j = 0; j < dimension; j++)
            tmp = fscanf(fin, "%f", &entityVec[last + j]);
    }
    fclose(fin);
    fin = fopen((initPath + "relation2vec" + note + ".vec").c_str(), "r");
    for (long i = 0; i < relationTotal; i++) {
        long last = i * dimensionR;
        for (long j = 0; j < dimensionR; j++)
            tmp = fscanf(fin, "%f", &relationVec[last + j]);
    }
    fclose(fin);
    fin = fopen((initPath + "A" + note + ".vec").c_str(), "r");
    for (long i = 0; i < relationTotal; i++)
            for (long jj = 0; jj < dimension; jj++)
                for (long ii = 0; ii < dimensionR; ii++)
                    tmp = fscanf(fin, "%f", &matrix[i * dimensionR * dimension + jj + ii * dimension]);
    fclose(fin);

    printf("init done\n");

    fin = fopen((loadPath + "M" + note + ".vec").c_str(), "r");

    for (long i = 0; i < relationTotal*2; i++) {
        long last = i * dimension*(dimension+1)/2;
        for (long j = 0; j < dimension*(dimension+1)/2; j++)
            tmp = fscanf(fin, "%f", &matrix_h[last + j]);
    }

    fclose(fin);

     fin = fopen((loadPath + "C" + note + ".vec").c_str(), "r");
    for (long i = 0; i < relationTotal*2; i++) {
        long last = i * dimension;
        for (long j = 0; j < dimension; j++)
            tmp = fscanf(fin, "%f", &centr_h[last + j]);
    }
 	fclose(fin);

 	printf("load done\n");

    entityRelVec = (float *)calloc(entityTotal * relationTotal * dimensionR,sizeof(float));
    pthread_t *pt = (pthread_t *)malloc(Threads * sizeof(pthread_t));
    for (long a = 0; a < Threads; a++)
        pthread_create(&pt[a], NULL, prepareMode,  (void*)a);
    for (long a = 0; a < Threads; a++)
        pthread_join(pt[a], NULL);
    free(pt);

}

float calc_sum(long e1, long e2, long rel) {
    float res = 0;
    long last1 = e1 * relationTotal * dimensionR + rel * dimensionR;
    long last2 = e2 * relationTotal * dimensionR + rel * dimensionR;
    long lastr = rel * dimensionR;
    for (long i = 0; i < dimensionR; i++)
        res += fabs(entityRelVec[last1 + i] + relationVec[lastr + i] - entityRelVec[last2 + i]);
    return res;
}

REAL cal_k_h(INT e,INT rel, REAL *M){
     // cout<<"line:  "<<242<<endl;
	INT lasta = e * relationTotal * dimensionR + rel * dimensionR;
	INT lastc = rel * dimension;
	REAL k=0;
	for(INT i=0;i<dimension;i++)
		for(INT j=i;j<dimension;j++){
			k+=2*M[i * dimension + j -  i*(i+1)/2]*(entityRelVec[lasta+i]-centr_h[lastc+i])*(entityRelVec[lasta+j]-centr_h[lastc+j]);
			//cout<<"d of k_h:  "<<2*M[i * dimension + j -  i*(i+1)/2]*(entityVec[lasta+i]-centr_h[lastc+i])*(entityVec[lasta+j]-centr_h[lastc+j])<<endl;
			//cout<<"k_h:  "<<k<<endl;
		}

	return k;
}
REAL cal_k_t(INT e,INT rel, REAL *M){
     // cout<<"line:  "<<256<<endl;
	INT lasta = e * relationTotal * dimensionR + rel * dimensionR;
	INT lastc = rel * dimension;
	REAL k=0;
	for(INT i=0;i<dimension;i++)
		for(INT j=i;j<dimension;j++)
			k+=2*M[i * dimension + j -  i*(i+1)/2]*(entityRelVec[lasta+i]-centr_t[lastc+i])*(entityRelVec[lasta+j]-centr_t[lastc+j]);
	return k;
}

REAL cal_y_h(INT e,INT rel){
  //  cout<<"line:  "<<267<<endl;
	INT lasta = e * relationTotal * dimensionR + rel * dimensionR;
	INT lastc =  rel *dimension;
	REAL y=0;
	for(INT i=0;i<dimension;i++)
		y+=(entityRelVec[lasta+i]-centr_h[lastc+i])*(entityRelVec[lasta+i]-centr_h[lastc+i]);
	return y;
}
REAL cal_y_t(INT e,INT rel){
	INT lasta = e * relationTotal * dimensionR + rel * dimensionR;
	INT lastc =  rel *dimension;
	REAL y=0;
	for(INT i=0;i<dimension;i++)
		y+=(entityRelVec[lasta+i]-centr_t[lastc+i])*(entityRelVec[lasta+i]-centr_t[lastc+i]);
	return y;
}
REAL cal_hEntDom(long eh,long rel)
{
	float *M_h=matrix_h+rel*(dimension+1)*dimension/2;
	REAL k_h=cal_k_h(eh, rel, M_h);
	REAL y_h=cal_y_h(eh, rel);
	REAL d_h;
	REAL tmp1=1-pow(k_h,-0.5);
	if(tmp1<0)
        d_h=0;
    else
        d_h=tmp1*tmp1*y_h;
	return d_h;
}
REAL cal_tEntDom(long et,long rel)
{
	float *M_t=matrix_t+rel*(dimension+1)*dimension/2;
    REAL k_t=cal_k_t(et, rel, M_t);
    REAL y_t=cal_y_t(et, rel);
	REAL d_t;
	REAL tmp2=1-pow(k_t,-0.5);
    if(tmp2<0)
        d_t=0;
    else
        d_t=tmp2*tmp2*y_t;
	return d_t;
}


//int i=0;
float calc_value(long eh,long et,long rel){
   // cout<<"cv\n"<<endl;
	float res=calc_sum(eh,et,rel);
	float d_h = 0,d_t = 0;
	if(domain){
		d_h = hEntDom[eh * relationTotal + rel];
		d_t = tEntDom[et * relationTotal + rel];
	}
	return res+(d_h+d_t);
}

bool find(long h, long t, long r) {
    long lef = 0;
    long rig = tripleTotal - 1;
    long mid;
    while (lef + 1 < rig) {
        long mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
}

float *l_filter_tot_10[6], *r_filter_tot_10[6], *l_tot_10[6], *r_tot_10[6];
float *l_filter_tot_3[6], *r_filter_tot_3[6], *l_tot_3[6], *r_tot_3[6];
float *l_filter_tot_1[6], *r_filter_tot_1[6], *l_tot_1[6], *r_tot_1[6];
float *l_filter_rank[6], *r_filter_rank[6], *l_rank[6], *r_rank[6];


void* testMode(void *con) {
	//cout<<"line 327\n";
    long id;
    id = (unsigned long long)(con);
    long lef = testTotal / (Threads) * id;
    long rig = testTotal / (Threads) * (id + 1) - 1;
    if (id == Threads - 1) rig = testTotal - 1;
    for (long i = lef; i <= rig; i++) {
        testid+=1;
        for(int j=1;j<=10;j++)
        {
            if(testid==testTotal/10*j)printf("%.2f\n",testid/testTotal);
        }
        long h = testList[i].h;
        long t = testList[i].t;
        long r = testList[i].r;
        long label = testList[i].label;
        float minimal = calc_value(h, t, r);
        long l_filter_s = 0;
        long l_s = 0;
        long r_filter_s = 0;
        long r_s = 0;
        long l_filter_s_constrain = 0;
        long l_s_constrain = 0;
        long r_filter_s_constrain = 0;
        long r_s_constrain = 0;
        long type_head = head_lef[r], type_tail = tail_lef[r];
        for (long j = 0; j <= entityTotal; j++) {
            if (j != h) {
            	//cout<<"line 349\n";
                float value = calc_value(j, t, r);
                if (value < minimal) {
                    l_s += 1;
                    if (not find(j, t, r))
                        l_filter_s += 1;
                }
                while (type_head < head_rig[r] && head_type[type_head] < j) type_head++;
                if (type_head < head_rig[r] && head_type[type_head] == j) {
                    if (value < minimal) {
                        l_s_constrain += 1;
                        if (not find(j, t, r))
                            l_filter_s_constrain += 1;
                    }
                }
            }
            if (j != t) {
                float value = calc_value(h, j, r);
                if (value < minimal) {
                    r_s += 1;
                    if (not find(h, j, r))
                        r_filter_s += 1;
                }
                while (type_tail < tail_rig[r] && tail_type[type_tail] < j) type_tail++;
                if (type_tail < tail_rig[r] && tail_type[type_tail] == j) {
                    if (value < minimal) {
                        r_s_constrain += 1;
                        if (not find(h, j, r))
                            r_filter_s_constrain += 1;
                    }
                }
            }
        }
        if (l_filter_s < 10) l_filter_tot_10[0][id] += 1;
        if (l_s < 10) l_tot_10[0][id] += 1;
        if (r_filter_s < 10) r_filter_tot_10[0][id] += 1;
        if (r_s < 10) r_tot_10[0][id] += 1;

        if (l_filter_s < 3) l_filter_tot_3[0][id] += 1;
        if (l_s < 3) l_tot_3[0][id] += 1;
        if (r_filter_s < 3) r_filter_tot_3[0][id] += 1;
        if (r_s < 3) r_tot_3[0][id] += 1;

        if (l_filter_s < 1) l_filter_tot_1[0][id] += 1;
        if (l_s < 1) l_tot_1[0][id] += 1;
        if (r_filter_s < 1) r_filter_tot_1[0][id] += 1;
        if (r_s < 1) r_tot_1[0][id] += 1;

        l_filter_rank[0][id] += l_filter_s;
        r_filter_rank[0][id] += r_filter_s;
        l_rank[0][id] += l_s;
        r_rank[0][id] += r_s;

        if (l_filter_s < 10) l_filter_tot_10[label][id] += 1;
        if (l_s < 10) l_tot_10[label][id] += 1;
        if (r_filter_s < 10) r_filter_tot_10[label][id] += 1;
        if (r_s < 10) r_tot_10[label][id] += 1;

        if (l_filter_s < 3) l_filter_tot_3[label][id] += 1;
        if (l_s < 3) l_tot_3[label][id] += 1;
        if (r_filter_s < 3) r_filter_tot_3[label][id] += 1;
        if (r_s < 3) r_tot_3[label][id] += 1;

        if (l_filter_s < 1) l_filter_tot_1[label][id] += 1;
        if (l_s < 1) l_tot_1[label][id] += 1;
        if (r_filter_s < 1) r_filter_tot_1[label][id] += 1;
        if (r_s < 1) r_tot_1[label][id] += 1;

        l_filter_rank[label][id] += l_filter_s;
        r_filter_rank[label][id] += r_filter_s;
        l_rank[label][id] += l_s;
        r_rank[label][id] += r_s;



        if (l_filter_s_constrain < 10) l_filter_tot_10[5][id] += 1;
        if (l_s_constrain < 10) l_tot_10[5][id] += 1;
        if (r_filter_s_constrain < 10) r_filter_tot_10[5][id] += 1;
        if (r_s_constrain < 10) r_tot_10[5][id] += 1;

        if (l_filter_s_constrain < 3) l_filter_tot_3[5][id] += 1;
        if (l_s_constrain < 3) l_tot_3[5][id] += 1;
        if (r_filter_s_constrain < 3) r_filter_tot_3[5][id] += 1;
        if (r_s_constrain < 3) r_tot_3[5][id] += 1;

        if (l_filter_s_constrain < 1) l_filter_tot_1[5][id] += 1;
        if (l_s_constrain < 1) l_tot_1[5][id] += 1;
        if (r_filter_s_constrain < 1) r_filter_tot_1[5][id] += 1;
        if (r_s_constrain < 1) r_tot_1[5][id] += 1;

        l_filter_rank[5][id] += l_filter_s_constrain;
        r_filter_rank[5][id] += r_filter_s_constrain;
        l_rank[5][id] += l_s_constrain;
        r_rank[5][id] += r_s_constrain;


    }
    pthread_exit(NULL);
}

void* test(void *con) {
    for (int i = 0; i <= 5; i++) {
        l_filter_tot_10[i] = (float *)calloc(Threads, sizeof(float));
        r_filter_tot_10[i] = (float *)calloc(Threads, sizeof(float));
        l_tot_10[i] = (float *)calloc(Threads, sizeof(float));
        r_tot_10[i] = (float *)calloc(Threads, sizeof(float));

        l_filter_tot_3[i] = (float *)calloc(Threads, sizeof(float));
        r_filter_tot_3[i] = (float *)calloc(Threads, sizeof(float));
        l_tot_3[i] = (float *)calloc(Threads, sizeof(float));
        r_tot_3[i] = (float *)calloc(Threads, sizeof(float));

        l_filter_tot_1[i] = (float *)calloc(Threads, sizeof(float));
        r_filter_tot_1[i] = (float *)calloc(Threads, sizeof(float));
        l_tot_1[i] = (float *)calloc(Threads, sizeof(float));
        r_tot_1[i] = (float *)calloc(Threads, sizeof(float));

        l_filter_rank[i] = (float *)calloc(Threads, sizeof(float));
        r_filter_rank[i] = (float *)calloc(Threads, sizeof(float));
        l_rank[i] = (float *)calloc(Threads, sizeof(float));
        r_rank[i] = (float *)calloc(Threads, sizeof(float));
    }
    pthread_t *pt = (pthread_t *)malloc(Threads * sizeof(pthread_t));
    for (long a = 0; a < Threads; a++)
        pthread_create(&pt[a], NULL, testMode,  (void*)a);
    for (long a = 0; a < Threads; a++)
        pthread_join(pt[a], NULL);
    free(pt);
    for (int i = 0; i <= 5; i++)
    for (long a = 1; a < Threads; a++) {
        l_filter_tot_10[i][a] += l_filter_tot_10[i][a - 1];
        r_filter_tot_10[i][a] += r_filter_tot_10[i][a - 1];
        l_tot_10[i][a] += l_tot_10[i][a - 1];
        r_tot_10[i][a] += r_tot_10[i][a - 1];

        l_filter_tot_3[i][a] += l_filter_tot_3[i][a - 1];
        r_filter_tot_3[i][a] += r_filter_tot_3[i][a - 1];
        l_tot_3[i][a] += l_tot_3[i][a - 1];
        r_tot_3[i][a] += r_tot_3[i][a - 1];

        l_filter_tot_1[i][a] += l_filter_tot_1[i][a - 1];
        r_filter_tot_1[i][a] += r_filter_tot_1[i][a - 1];
        l_tot_1[i][a] += l_tot_1[i][a - 1];
        r_tot_1[i][a] += r_tot_1[i][a - 1];

        l_filter_rank[i][a] += l_filter_rank[i][a - 1];
        r_filter_rank[i][a] += r_filter_rank[i][a - 1];
        l_rank[i][a] += l_rank[i][a - 1];
        r_rank[i][a] += r_rank[i][a - 1];
    }
    printf("10:\n");
    for (int i = 0; i <= 0; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / testTotal, l_tot_10[i][Threads - 1] / testTotal);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / testTotal, l_filter_tot_10[i][Threads - 1] / testTotal);
        printf("right %f %f\n", r_rank[i][Threads - 1] / testTotal, r_tot_10[i][Threads - 1] / testTotal);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / testTotal, r_filter_tot_10[i][Threads - 1] / testTotal);
    }
    for (int i = 5; i <= 5; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / testTotal, l_tot_10[i][Threads - 1] / testTotal);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / testTotal, l_filter_tot_10[i][Threads - 1] / testTotal);
        printf("right %f %f\n", r_rank[i][Threads - 1] / testTotal, r_tot_10[i][Threads - 1] / testTotal);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / testTotal, r_filter_tot_10[i][Threads - 1] / testTotal);
    }
    for (int i = 1; i <= 4; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / nntotal[i], l_tot_10[i][Threads - 1] / nntotal[i]);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / nntotal[i], l_filter_tot_10[i][Threads - 1] / nntotal[i]);
        printf("right %f %f\n", r_rank[i][Threads - 1] / nntotal[i], r_tot_10[i][Threads - 1] / nntotal[i]);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / nntotal[i], r_filter_tot_10[i][Threads - 1] / nntotal[i]);
    }

    printf("3:\n");
    for (int i = 0; i <= 0; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / testTotal, l_tot_3[i][Threads - 1] / testTotal);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / testTotal, l_filter_tot_3[i][Threads - 1] / testTotal);
        printf("right %f %f\n", r_rank[i][Threads - 1] / testTotal, r_tot_3[i][Threads - 1] / testTotal);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / testTotal, r_filter_tot_3[i][Threads - 1] / testTotal);
    }
    for (int i = 5; i <= 5; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / testTotal, l_tot_3[i][Threads - 1] / testTotal);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / testTotal, l_filter_tot_3[i][Threads - 1] / testTotal);
        printf("right %f %f\n", r_rank[i][Threads - 1] / testTotal, r_tot_3[i][Threads - 1] / testTotal);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / testTotal, r_filter_tot_3[i][Threads - 1] / testTotal);
    }
    for (int i = 1; i <= 4; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / nntotal[i], l_tot_3[i][Threads - 1] / nntotal[i]);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / nntotal[i], l_filter_tot_3[i][Threads - 1] / nntotal[i]);
        printf("right %f %f\n", r_rank[i][Threads - 1] / nntotal[i], r_tot_3[i][Threads - 1] / nntotal[i]);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / nntotal[i], r_filter_tot_3[i][Threads - 1] / nntotal[i]);
    }

    printf("1:\n");
    for (int i = 0; i <= 0; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / testTotal, l_tot_1[i][Threads - 1] / testTotal);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / testTotal, l_filter_tot_1[i][Threads - 1] / testTotal);
        printf("right %f %f\n", r_rank[i][Threads - 1] / testTotal, r_tot_1[i][Threads - 1] / testTotal);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / testTotal, r_filter_tot_1[i][Threads - 1] / testTotal);
    }
    for (int i = 5; i <= 5; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / testTotal, l_tot_1[i][Threads - 1] / testTotal);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / testTotal, l_filter_tot_1[i][Threads - 1] / testTotal);
        printf("right %f %f\n", r_rank[i][Threads - 1] / testTotal, r_tot_1[i][Threads - 1] / testTotal);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / testTotal, r_filter_tot_1[i][Threads - 1] / testTotal);
    }
    for (int i = 1; i <= 4; i++) {
        printf("left %f %f\n", l_rank[i][Threads - 1] / nntotal[i], l_tot_1[i][Threads - 1] / nntotal[i]);
        printf("left(filter) %f %f\n", l_filter_rank[i][Threads - 1] / nntotal[i], l_filter_tot_1[i][Threads - 1] / nntotal[i]);
        printf("right %f %f\n", r_rank[i][Threads - 1] / nntotal[i], r_tot_1[i][Threads - 1] / nntotal[i]);
        printf("right(filter) %f %f\n", r_filter_rank[i][Threads - 1] / nntotal[i], r_filter_tot_1[i][Threads - 1] / nntotal[i]);
    }
}


long ArgPos(char *str, long argc, char **argv) {
    long a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

void setparameters(int argc, char **argv) {
    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sizeR", argc, argv)) > 0) dimensionR = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-input", argc, argv)) > 0) inPath = argv[i + 1];
    if ((i = ArgPos((char *)"-init", argc, argv)) > 0) initPath = argv[i + 1];
    if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) Threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-load", argc, argv)) > 0) loadPath = argv[i + 1];
    if ((i = ArgPos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
	if ((i = ArgPos((char *)"-domain", argc, argv)) > 0) domain = atoi(argv[i + 1]);
}

int main(int argc, char **argv) {
    setparameters(argc, argv);
    init();
    prepare();
    test(NULL);
    return 0;
}
