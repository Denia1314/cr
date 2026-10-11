/* Experimental stable-baseline differences; no target writes or memory dumps. */
#define main fixture_main
#include "memory_lab.c"
#undef main
#define CHUNK 65536
#define LIMIT (512ULL*1024*1024)
#define MAX_CANDIDATES 20000
typedef struct {uint64_t address;size_t length;unsigned char *bytes,*stable;} Region;
typedef struct {uint64_t address,before;} Candidate;
static Candidate candidates[MAX_CANDIDATES];
static unsigned total;
static bool stopped;
static bool poll_stop(void) {
    fd_set descriptors;FD_ZERO(&descriptors);FD_SET(STDIN_FILENO,&descriptors);
    struct timeval timeout={0,0};
    if(select(1,&descriptors,NULL,NULL,&timeout)==1){char command;ssize_t n=read(STDIN_FILENO,&command,1);if(n==0||(n==1&&command=='s'))return true;}
    return false;
}
static bool wait_trigger(void) {
    fd_set descriptors;FD_ZERO(&descriptors);FD_SET(STDIN_FILENO,&descriptors);
    struct timeval timeout={90,0};
    char command=0;
    return select(1,&descriptors,NULL,NULL,&timeout)==1&&read(STDIN_FILENO,&command,1)==1&&command=='\n';
}
static void emit_sample(int mem,uint64_t begun,unsigned previous_total,bool truncated) {
    printf("{\"event\":\"raw_sample\",\"at_ns\":%"PRIu64",\"values\":[",begun);
    for(unsigned i=0;i<total;++i){uint64_t bits=0;if(i)putchar(',');
        if(pread(mem,&bits,8,(off_t)candidates[i].address)==8)printf("\"%"PRIx64"\"",bits);else printf("null");
    }printf("],\"new_rows\":[");
    for(unsigned i=previous_total;i<total;++i)printf("%s[\"%"PRIx64"\",\"%"PRIx64"\"]",i==previous_total?"":",",candidates[i].address,candidates[i].before);
    printf("],\"discovery_truncated\":%s,\"read_end_ns\":%"PRIu64"}\n",truncated?"true":"false",clock_ns());fflush(stdout);
}
static void emit_footer(uint64_t failed,bool truncated) {
    printf("{\"event\":\"raw_candidates\",\"stopped\":%s,\"discovery_truncated\":%s,\"failed_bytes\":%"PRIu64",\"rows\":[",stopped?"true":"false",truncated?"true":"false",failed);
    for(unsigned i=0;i<total;++i)printf("%s[\"%"PRIx64"\",\"%"PRIx64"\"]",i?",":"",candidates[i].address,candidates[i].before);
    puts("]}");fflush(stdout);
}
static uint64_t raw8(const unsigned char *p){uint64_t v;memcpy(&v,p,8);return v;}
static void mark_stable(Region *r,const unsigned char *current) {
    for(size_t i=0;i+8<=r->length;i+=4)r->stable[i/4]=(memcmp(r->bytes+i,current+i,8)==0);
    memcpy(r->bytes,current,r->length);
}
static bool discover(Region *r,const unsigned char *current) {
    for(size_t i=0;i+8<=r->length;i+=4)if(r->stable[i/4]&&memcmp(r->bytes+i,current+i,8)!=0) {
        if(total==MAX_CANDIDATES)return false;
        candidates[total++]=(Candidate){r->address+i,raw8(r->bytes+i)};
        r->stable[i/4]=0;
    }
    return true;
}
static int private_test(void) {
    uint64_t *remote=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(remote==MAP_FAILED)return 2;
    remote[0]=10000000;remote[1]=10000000;remote[2]=0;
    int command[2],ack[2];if(pipe(command)||pipe(ack))return 2;
    pid_t child=fork();if(child<0)return 2;
    if(!child){close(command[1]);close(ack[0]);unsigned phase;
        while(transfer(command[0],&phase,sizeof(phase),false)&&phase){
            remote[0]=phase<3?10000000:phase==3?7000000:8000000;
            remote[1]=10000000;remote[2]=phase;
            if(!transfer(ack[1],&phase,sizeof(phase),true))break;
        }_exit(0);
    }
    close(command[0]);close(ack[1]);unsigned char first[24],current[24],stable[6]={0};
    Region r={(uint64_t)(uintptr_t)remote,24,first,stable};bool ok=true;
    for(unsigned phase=1;phase<=4&&ok;++phase){unsigned reply=0;
        ok=transfer(command[1],&phase,sizeof(phase),true)&&transfer(ack[0],&reply,sizeof(reply),false)&&reply==phase;
        ok=ok&&read_child(child,remote,current,sizeof(current))==(ssize_t)sizeof(current);
        if(phase==1)memcpy(first,current,sizeof(first));
        else if(phase==2)mark_stable(&r,current);
        else ok=ok&&discover(&r,current);
        if(phase==3)ok=ok&&total==1&&candidates[0].address==(uint64_t)(uintptr_t)remote&&candidates[0].before==10000000&&raw8(current)==7000000;
        if(phase==4)ok=ok&&total==1&&raw8(current)==8000000;
    }
    /* The unchanged constant and changing control counter must not become candidates. */
    ok=ok&&stable[4]==0; /* continuously changing control counter excluded */
    unsigned accepted=total;total=MAX_CANDIDATES;stable[0]=1;
    ok=ok&&!discover(&r,current)&&total==MAX_CANDIDATES;total=accepted;
    unsigned stop=0;(void)transfer(command[1],&stop,sizeof(stop),true);close(command[1]);close(ack[0]);
    int status=0;waitpid(child,&status,0);ok=ok&&WIFEXITED(status)&&WEXITSTATUS(status)==0;
    munmap(remote,4096);
    printf("{\"event\":\"stable_fixture\",\"passed\":%s,\"source\":\"private_child\",\"candidate_windows\":%u,\"game_fields_verified\":false}\n",ok?"true":"false",total);
    return ok?0:1;
}
static int overflow_fixture(bool gated) {
    size_t length=(MAX_CANDIDATES+2)*8;
    uint64_t *remote=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(remote==MAP_FAILED)return 2;remote[0]=10000000;
    int command[2],ack[2];if(pipe(command)||pipe(ack))return 2;pid_t child=fork();if(child<0)return 2;
    if(!child){close(command[1]);close(ack[0]);unsigned phase;
        while(transfer(command[0],&phase,sizeof(phase),false)&&phase){
            if(phase>=3){uint64_t gain=(uint64_t)(phase-3)*200000/28*10;remote[0]=7000000+gain;if(remote[0]>10000000)remote[0]=10000000;
                for(unsigned j=1;j<MAX_CANDIDATES+2;++j)remote[j]=1234;}
            if(!transfer(ack[1],&phase,sizeof(phase),true))break;
        }_exit(0);
    }
    close(command[0]);close(ack[1]);char path[64];snprintf(path,sizeof(path),"/proc/%d/mem",child);int mem=open(path,O_RDONLY|O_CLOEXEC);
    unsigned char *first=malloc(length),*current=malloc(length),*stable=calloc(length/4,1);if(mem<0||!first||!current||!stable){kill(child,SIGTERM);waitpid(child,NULL,0);return 2;}
    Region region={(uint64_t)(uintptr_t)remote,length,first,stable};uint64_t session=start_ticks(child);bool ok=session!=0,truncated=false;
    for(unsigned phase=1;phase<=52&&ok;++phase){unsigned reply=0;
        ok=transfer(command[1],&phase,sizeof(phase),true)&&transfer(ack[0],&reply,sizeof(reply),false)&&reply==phase;
        ok=ok&&pread(mem,current,length,(off_t)(uintptr_t)remote)==(ssize_t)length&&start_ticks(child)==session;
        if(!ok)break;
        if(phase==1)memcpy(first,current,length);
        else if(phase==2){mark_stable(&region,current);printf("{\"event\":\"stable_ready\",\"at_ns\":%"PRIu64",\"source\":\"private_overflow_fixture\",\"cap\":%u}\n",clock_ns(),MAX_CANDIDATES);fflush(stdout);
            if(gated){ok=wait_trigger()&&total==0;if(!ok)break;
                printf("{\"event\":\"discovery_started\",\"at_ns\":%"PRIu64",\"candidates_before_trigger\":%u}\n",clock_ns(),total);fflush(stdout);}}
        else {unsigned previous_total=total;
            if(!truncated&&!discover(&region,current))truncated=true;
            emit_sample(mem,clock_ns(),previous_total,truncated);
            ok=ok&&truncated&&total==MAX_CANDIDATES&&candidates[0].address==(uint64_t)(uintptr_t)remote;
            struct timespec pause={0,200000000L};nanosleep(&pause,NULL);
        }
    }
    unsigned stop=0;(void)transfer(command[1],&stop,sizeof(stop),true);close(command[1]);close(ack[0]);int status=0;waitpid(child,&status,0);
    ok=ok&&WIFEXITED(status)&&WEXITSTATUS(status)==0&&raw8(current)==10000000;
    if(ok)emit_footer(0,truncated);
    close(mem);free(first);free(current);free(stable);munmap(remote,length);return ok?0:1;
}
int main(int argc,char **argv) {
    if(argc==2&&strcmp(argv[1],"clock")==0){printf("%"PRIu64"\n",clock_ns());return 0;}
    if(argc==2&&strcmp(argv[1],"self-test")==0)return private_test();
    if(argc==2&&strcmp(argv[1],"overflow-fixture")==0)return overflow_fixture(false);
    if(argc==2&&strcmp(argv[1],"trigger-fixture")==0)return overflow_fixture(true);
    bool gated=argc==4&&strcmp(argv[3],"triggered")==0;
    uint64_t pid,started;
    if((argc!=3&&!gated)||!integer(argv[1],&pid)||!integer(argv[2],&started)||pid>INT32_MAX||start_ticks((pid_t)pid)!=started)return 64;
    char path[64];snprintf(path,sizeof(path),"/proc/%u/mem",(unsigned)pid);int mem=open(path,O_RDONLY|O_CLOEXEC);
    snprintf(path,sizeof(path),"/proc/%u/maps",(unsigned)pid);FILE *maps=fopen(path,"r");
    if(mem<0||!maps)return 2;
    Region *regions=calloc(LIMIT/CHUNK+1,sizeof(Region));if(!regions)return 2;
    unsigned count=0;uint64_t attempted=0,failed=0,eligible=0,baseline_begin=clock_ns();char line[2048],permissions[5];
    while(fgets(line,sizeof(line),maps)) {
        unsigned long long begin,end;int at=0;
        if(sscanf(line,"%llx-%llx %4s %*s %*s %*s %n",&begin,&end,permissions,&at)!=3)continue;
        const char *name=line+at;bool named=strstr(name,"[anon:Mem_")!=NULL;
        if(!(named||name[0]=='\n'||name[0]==0)||permissions[0]!='r'||permissions[1]!='w'||(permissions[2]=='x'&&!named)||end<=begin)continue;
        eligible+=end-begin;
        for(uint64_t pos=begin;pos<end&&attempted<LIMIT;pos+=CHUNK){
            size_t len=end-pos<CHUNK?(size_t)(end-pos):CHUNK;if(len>LIMIT-attempted)len=(size_t)(LIMIT-attempted);attempted+=len;
            unsigned char *bytes=malloc(len),*stable=calloc(len/4,1);
            if(!bytes||!stable){free(bytes);free(stable);return 2;}
            if(pread(mem,bytes,len,(off_t)pos)!=(ssize_t)len){free(bytes);free(stable);failed+=len;continue;}
            regions[count++]=(Region){pos,len,bytes,stable};
        }
    }fclose(maps);
    struct timespec delay={2,0};nanosleep(&delay,NULL);unsigned char buffer[CHUNK];
    if(start_ticks((pid_t)pid)!=started)return 2;
    for(unsigned j=0;j<count;++j){Region *r=&regions[j];
        if(pread(mem,buffer,r->length,(off_t)r->address)!=(ssize_t)r->length){failed+=r->length;memset(r->stable,0,r->length/4);continue;}
        mark_stable(r,buffer);
    }
    printf("{\"event\":\"stable_ready\",\"bytes\":%"PRIu64",\"eligible_bytes\":%"PRIu64",\"failed_bytes\":%"PRIu64",\"baseline_begin_ns\":%"PRIu64",\"at_ns\":%"PRIu64",\"cap\":%u}\n",attempted,eligible,failed,baseline_begin,clock_ns(),MAX_CANDIDATES);fflush(stdout);
    if(gated){
        if(!wait_trigger()||start_ticks((pid_t)pid)!=started)return 4;
        /* Remove waiting-period noise before user action, without discovering candidates. */
        for(unsigned j=0;j<count;++j){Region *r=&regions[j];
            if(pread(mem,buffer,r->length,(off_t)r->address)!=(ssize_t)r->length){failed+=r->length;memset(r->stable,0,r->length/4);continue;}
            for(size_t i=0;i+8<=r->length;i+=4)r->stable[i/4]=r->stable[i/4]&&memcmp(r->bytes+i,buffer+i,8)==0;
            memcpy(r->bytes,buffer,r->length);
        }
        printf("{\"event\":\"discovery_started\",\"at_ns\":%"PRIu64",\"candidates_before_trigger\":%u}\n",clock_ns(),total);fflush(stdout);
    }
    bool discovery_truncated=false;
    for(unsigned tick=0;tick<120;++tick){
        if(poll_stop()){stopped=true;break;}
        if(start_ticks((pid_t)pid)!=started)return 2;uint64_t begun=clock_ns();
        unsigned previous_total=total;
        if(tick<60&&!discovery_truncated)for(unsigned j=0;j<count;++j){Region *r=&regions[j];
            if(pread(mem,buffer,r->length,(off_t)r->address)!=(ssize_t)r->length){failed+=r->length;continue;}
            if(memcmp(buffer,r->bytes,r->length)!=0&&!discover(r,buffer)){discovery_truncated=true;break;}
        }
        emit_sample(mem,begun,previous_total,discovery_truncated);
        delay=(struct timespec){0,500000000L};nanosleep(&delay,NULL);
    }
    if(start_ticks((pid_t)pid)!=started)return 2;
    emit_footer(failed,discovery_truncated);for(unsigned j=0;j<count;++j){free(regions[j].bytes);free(regions[j].stable);}free(regions);close(mem);return 0;
}

