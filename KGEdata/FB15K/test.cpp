#include<cstdio>
using namespace std;
int main()
{
    /*FILE *fin;
    fin=fopen("test2id_all.txt","r");
    int in;
    int k[4]={0,0,0,0};
    int num;
    int tmp=fscanf(fin,"%d",&num);
    printf("%d\n",num);
    for(int i=0;i<num;i++)
    {
        tmp=fscanf(fin,"%d%d%d%d",&in,&tmp,&tmp,&tmp);
        k[in]++;
    }
    printf("%d %d %d %d\n",k[0],k[1],k[2],k[3]);
    return 0;*/

    FILE *f1,*f2;
    f1 = fopen("test2id.txt","r");
    f2 = fopen("train2id.txt","r");
    int tmp,num;
    int k=0;
    tmp=fscanf(f1,"%d",&num);
    for(int i=0;i<num;i++)
    {
        int rel;
        tmp=fscanf(f1,"%d%d%d",&tmp,&tmp,&rel);
        if(rel==95)k++;
    }
    printf("%d\n",k);
    k=0;
    tmp=fscanf(f2,"%d",&num);
    for(int i=0;i<num;i++)
    {
        int rel;
        tmp=fscanf(f2,"%d%d%d",&tmp,&tmp,&rel);
        if(rel==95)k++;
    }
    printf("%d\n",k);
    return 0;
}
