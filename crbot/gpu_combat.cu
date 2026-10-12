// Reference-order detailed combat. No fast-math; independent branch per thread.
#define INFINITY __longlong_as_double(0x7ff0000000000000ULL)
#define E(i,f) e[(i)*NF+(f)]
__device__ double distance(double* e,int a,int b) {
 return fmax(0.,hypot(E(a,X)-E(b,X),E(a,Y)-E(b,Y))-E(a,RADIUS)-E(b,RADIUS));
}
__device__ unsigned long long timer() { unsigned long long v; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)); return v; }
__device__ int cell(double x,double y) {return max(0,min(17,(int)x))*32+max(0,min(31,(int)y));}
__device__ bool pass(int x,int y,bool* blocked,double b0,double b1) {
 return x>=0 && x<18 && y>=0 && y<32 && !blocked[x*32+y] &&
  ((y!=15 && y!=16)||fabs(x+.5-b0)<=1 ||fabs(x+.5-b1)<=1);
}
__device__ bool less(double p,int n,double q,int m) {return p<q || (p==q && n<m);}
__device__ bool waypoint(double* e,int count,int a,int t,double b0,double b1,double &tx,double &ty,
 int* cacheStart,int* cacheGoal,unsigned long long* cacheHash,int* cacheNext,
 unsigned long long* routeKeys,int* routeValues) {
 tx=E(t,X);ty=E(t,Y);
 if(E(a,AIR))return true;
 int start=cell(E(a,X),E(a,Y)),goal=cell(tx,ty);
 unsigned long long hash=1469598103934665603ULL;
 for(int i=0;i<count;i++)if(i!=a && i!=t && E(i,HP)>0 && (E(i,TOWER)||E(i,BUILDING))) {
  hash=(hash ^ __double_as_longlong(E(i,X)))*1099511628211ULL;
  hash=(hash ^ __double_as_longlong(E(i,Y)))*1099511628211ULL;
  hash=(hash ^ __double_as_longlong(E(i,RADIUS)+fmin(.4,E(a,RADIUS))))*1099511628211ULL;
 }
 // Only A* cell routes are cached. The direct-line test depends on subcell
 // position and must still be repeated before accepting a cached cell route.
 bool blocked[576]={false};
 for(int i=0;i<count;i++) if(i!=a && i!=t && E(i,HP)>0 && (E(i,TOWER)||E(i,BUILDING))) {
  double r=E(i,RADIUS)+fmin(.4,E(a,RADIUS));
  for(int x=max(0,(int)(E(i,X)-r));x<min(18,(int)(E(i,X)+r)+1);x++)
   for(int y=max(0,(int)(E(i,Y)-r));y<min(32,(int)(E(i,Y)+r)+1);y++)
    if(hypot(x+.5-E(i,X),y+.5-E(i,Y))<r)blocked[x*32+y]=true;
 }
 if((E(a,Y)-16)*(ty-16)>=0) {
  int steps=max(1,(int)ceil(hypot(tx-E(a,X),ty-E(a,Y))*3));bool direct=true;
  for(int j=1;j<=steps;j++) {
   int c=cell(E(a,X)+(tx-E(a,X))*j/steps,E(a,Y)+(ty-E(a,Y))*j/steps);
   if(!pass(c/32,c%32,blocked,b0,b1)){direct=false;break;}
  }
  if(direct)return true;
 }
 if(cacheStart[a]==start && cacheGoal[a]==goal && cacheHash[a]==hash) {
  int next=cacheNext[a];if(next<0)return false;
  if(next!=goal){tx=next/32+.5;ty=next%32+.5;}return true;
 }
 unsigned long long key=(hash ^ (unsigned long long)(start*576+goal))*1099511628211ULL;
 key=(key ^ __double_as_longlong(b0))*1099511628211ULL;
 key=(key ^ __double_as_longlong(b1))*1099511628211ULL;if(!key)key=1;
 int slot=-1;
 for(int probe=0;probe<16;probe++) {
  int at=(int)((key+probe)&65535ULL);
  unsigned long long old=atomicCAS(routeKeys+at,0ULL,key);
  if(old==0 || old==key){slot=at;break;}
 }
 if(slot>=0) {
  int next=atomicAdd(routeValues+slot,0);
  if(next!=-3) {
   cacheStart[a]=start;cacheGoal[a]=goal;cacheHash[a]=hash;cacheNext[a]=next;
   if(next<0)return false;
   if(next!=goal){tx=next/32+.5;ty=next%32+.5;}return true;
  }
 }
 blocked[start]=blocked[goal]=false;
 double costs[576],priority[4096];int parent[576],nodes[4096];
 for(int j=0;j<576;j++){costs[j]=INFINITY;parent[j]=-1;}costs[start]=0;
 nodes[0]=start;priority[0]=0;int size=1;bool found=false;
 const int dx[8]={1,-1,0,0,1,1,-1,-1},dy[8]={0,0,1,-1,1,-1,1,-1};
 while(size) {
  int here=nodes[0];size--;
  nodes[0]=nodes[size];priority[0]=priority[size];int at=0;
  while(2*at+1<size) {
   int child=2*at+1;
   if(child+1<size && less(priority[child+1],nodes[child+1],priority[child],nodes[child]))child++;
   if(!less(priority[child],nodes[child],priority[at],nodes[at]))break;
   double pv=priority[at];priority[at]=priority[child];priority[child]=pv;
   int nv=nodes[at];nodes[at]=nodes[child];nodes[child]=nv;at=child;
  }
  if(here==goal){found=true;break;}
  int hx=here/32,hy=here%32;
  for(int j=0;j<8;j++) {
   int x=hx+dx[j],y=hy+dy[j];
   if(!pass(x,y,blocked,b0,b1))continue;
   if(dx[j] && dy[j] && (!pass(hx+dx[j],hy,blocked,b0,b1)||!pass(hx,hy+dy[j],blocked,b0,b1)))continue;
   int next=x*32+y;double cost=costs[here]+hypot((double)dx[j],(double)dy[j]);
   if(cost>=costs[next])continue;
   costs[next]=cost;parent[next]=here;
   if(size>=4096){cacheStart[a]=-2;return false;}
   at=size++;nodes[at]=next;priority[at]=cost+hypot((double)(x-goal/32),(double)(y-goal%32));
   while(at>0) {
    int p=(at-1)/2;if(!less(priority[at],nodes[at],priority[p],nodes[p]))break;
    double pv=priority[at];priority[at]=priority[p];priority[p]=pv;
    int nv=nodes[at];nodes[at]=nodes[p];nodes[p]=nv;at=p;
   }
  }
 }
 cacheStart[a]=start;cacheGoal[a]=goal;cacheHash[a]=hash;
 if(!found){cacheNext[a]=-1;if(slot>=0)atomicExch(routeValues+slot,-1);return false;}
 int next=goal;
 while(parent[next]>=0 && parent[next]!=start)next=parent[next];
 cacheNext[a]=next;
 if(slot>=0)atomicExch(routeValues+slot,next);
 if(next!=goal){tx=next/32+.5;ty=next%32+.5;}
 return true;
}
__device__ void control(double* e,int i,double time,double stun,double push,double x,double y) {
 if(stun) {E(i,STUNNED_UNTIL)=fmax(E(i,STUNNED_UNTIL),time+stun);E(i,WALKED)=0;E(i,LOCKED_AT)=time+stun;E(i,WINDING_TARGET)=-1;if(E(i,DASH_PHASE)==1)E(i,DASH_PHASE)=0;}
 if(push && !E(i,IGNORE_PUSHBACK) && !E(i,BUILDING) && !E(i,TOWER)) {
  double dx=E(i,X)-x,dy=E(i,Y)-y,d=hypot(dx,dy);
  if(d){E(i,X)=fmax(0.,fmin(18.,E(i,X)+dx/d*push));E(i,Y)=fmax(0.,fmin(32.,E(i,Y)+dy/d*push));}
  E(i,WALKED)=0;E(i,WINDING_TARGET)=-1;if(E(i,DASH_PHASE)==1)E(i,DASH_PHASE)=0;
 }
}
// EXTENDED_MECHANICS
extern "C" __global__ void advance_battles(double* entities,double* impacts,double* metadata,
 double* allEffects,double* effectScratch,double* prototypes,double* pulses,unsigned char* spawned,int prototypeCount,int entityCapacity,int effectCapacity,int* codes,
 unsigned long long* routeKeys,int* routeValues,
 int batches,double duration,double step,int grid,double b0,double b1,unsigned long long budget) {
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=batches)return;
 double* e=entities+k*entityCapacity*NF;double* events=impacts+k*NI*13;double* m=metadata+k*13;
 CombatContext c{e,events,allEffects+k*effectCapacity*EF,effectScratch+k*effectCapacity*EF,m,prototypes,pulses,spawned+k*prototypeCount,codes+k,(int)m[1],(int)m[2],(int)m[12],entityCapacity,effectCapacity};
 int count=(int)m[1],pending=(int)m[2];double end=m[0]+fmax(0.,duration);
 unsigned long long began=timer();
 int cacheStart[NE],cacheGoal[NE],cacheNext[NE];unsigned long long cacheHash[NE];
 for(int j=0;j<NE;j++)cacheStart[j]=-1;
 while(m[0]<end-1e-6) {
  if(timer()-began>=budget){codes[k]=1;return;}
  double dt=fmin(step,end-m[0]);m[0]+=dt;double time=m[0];
  c.count=count;c.pending=pending;
  if(c.effects)effects(c);
  if(*c.code)return;
  count=c.count;pending=c.pending;
  m[3]=fmin(10.,m[3]+dt/m[9]);m[4]=fmin(10.,m[4]+dt/m[9]);
  int retained=0;
  for(int j=0;j<pending;j++) {
   double* p=events+j*13;
   if(p[0]>time) {if(retained!=j)for(int f=0;f<13;f++)events[retained*13+f]=p[f];retained++;continue;}
   for(int i=0;i<count;i++) {
    if(E(i,SIDE)==p[1]||E(i,HP)<=0 ||(E(i,AIR) && !p[7]))continue;
    if(!((p[6]>0 && hypot(E(i,X)-p[3],E(i,Y)-p[4])<=p[6]+E(i,RADIUS)) ||(p[6]==0 && E(i,UID)==p[2])))continue;
    double damage=p[5]*(E(i,TOWER)?p[8]:1.);
    int attacker=by_uid(c,p[11]);
    if(!hit(c,i,damage,attacker))continue;
    control(e,i,time,p[9],p[10],p[3],p[4]+p[1]*.001);
   }
  }
  pending=retained;
  c.pending=pending;int turnCount=count;
  for(int a=0;a<turnCount;a++) {
   if(E(a,HP)<=0)continue;
   if(E(a,EXPIRES_AT)<=time){E(a,HP)=0;continue;}
   if(time<fmax(E(a,ACTIVE_AT),E(a,STUNNED_UNTIL)))continue;
   if(time>=E(a,RESOURCE_AT)) {
    m[E(a,SIDE)==1?3:4]=fmin(10.,m[E(a,SIDE)==1?3:4]+E(a,RESOURCE_AMOUNT));
    E(a,RESOURCE_AT)=time+E(a,RESOURCE_PERIOD);
   }
   if(E(a,SPAWN_AT)<=time && c.count<96) {
    if(E(a,SPAWN_SPEC)>=0)for(int j=0;j<min(8,(int)E(a,SPAWN_COUNT));j++)spawn(c,(int)E(a,SPAWN_SPEC),E(a,SIDE),E(a,X)+j*.3,E(a,Y),true,E(a,CLONED));
    E(a,SPAWN_AT)=time+fmax(1.,E(a,SPAWN_PERIOD));count=c.count;
    if(*c.code)return;
   }
   if(E(a,KING) && !E(a,ACTIVE)) {
    int princess=0;for(int j=0;j<count;j++)if(E(j,UID)>=0 && E(j,TOWER)&&E(j,SIDE)==E(a,SIDE)&&!E(j,KING))princess++;
    E(a,ACTIVE)=E(a,HP)<E(a,SPEC_HP)||princess<2;if(!E(a,ACTIVE))continue;
   }
   double sight=E(a,TOWER)?fmax(E(a,SIGHT),E(a,REACH)):E(a,SIGHT);
   int locked=-1,nearby=-1,tower=-1,interceptor=-1;double nearD=INFINITY,towerD=INFINITY;
   for(int j=0;j<count;j++) {
    if(E(j,SIDE)==E(a,SIDE)||E(j,HP)<=0 ||(E(j,DASH_IMMUNE)&&E(j,DASH_PHASE)==2))continue;
    if(E(j,AIR)?!E(a,AIR_TARGET):!E(a,GROUND_TARGET))continue;
    if(E(a,BUILDING_ONLY) && !E(j,BUILDING) && !E(j,TOWER))continue;
    double d=distance(e,a,j);if(d<E(a,MINIMUM_RANGE))continue;
    if(E(a,TOWER) && d>E(a,REACH))continue;
    if(E(j,UID)==E(a,TARGET)&&d<=sight)locked=j;
    if(d<=sight && d<nearD){nearby=j;nearD=d;}
    if(E(j,TOWER)&&d<towerD){tower=j;towerD=d;}
   }
   if(locked>=0 && E(locked,TOWER) && !E(a,TOWER) && E(a,BUILDING_ONLY) && distance(e,a,locked)>E(a,REACH) && E(a,WINDING_TARGET)<0) {
    double best=distance(e,a,locked);
    for(int j=0;j<count;j++)if(E(j,SIDE)!=E(a,SIDE) && E(j,HP)>0 && !(E(j,DASH_IMMUNE)&&E(j,DASH_PHASE)==2) && E(j,BUILDING)&&!E(j,TOWER) &&
      (E(j,AIR)?E(a,AIR_TARGET):E(a,GROUND_TARGET))) {
     double d=distance(e,a,j);if(d>=E(a,MINIMUM_RANGE) && d<=sight && d<best){best=d;interceptor=j;}
    }
   }
   int t=interceptor>=0?interceptor:locked>=0?locked:nearby>=0?nearby:tower;
   c.pending=pending;
   if(dash(c,a,t)){pending=c.pending;if(*c.code)return;continue;}
   if(t>=0 && E(t,UID)!=E(a,TARGET)){E(a,TARGET)=E(t,UID);E(a,WINDING_TARGET)=-1;}
   if(t<0)continue;
   double d=distance(e,a,t);
   if(d<=E(a,REACH)) {
    if(E(a,WINDING_TARGET)!=E(t,UID)){E(a,WINDING_TARGET)=E(t,UID);E(a,LOCKED_AT)=time;E(a,READY_AT)=fmax(E(a,READY_AT),time+E(a,FIRST_HIT));}
    if(time<E(a,READY_AT)||E(a,DAMAGE)<=0)continue;
    double damage=E(a,DAMAGE);
    if(E(a,CHARGE_DISTANCE) && E(a,WALKED)>=E(a,CHARGE_DISTANCE))damage=E(a,CHARGE_DAMAGE)?E(a,CHARGE_DAMAGE):damage;
    else if(E(a,RAMP_TIME0)) {
     double held=time-E(a,LOCKED_AT);
     if(held>=E(a,RAMP_TIME0)+E(a,RAMP_TIME1))damage=E(a,RAMP_DAMAGE1);
     else if(held>=E(a,RAMP_TIME0))damage=E(a,RAMP_DAMAGE0);
    }
    E(a,WALKED)=0;
    int victims[32],nv=1;victims[0]=t;
    if(E(a,CHAIN_COUNT)>1) {
     while(nv<(int)E(a,CHAIN_COUNT)) {
      int previous=victims[nv-1],chosen=-1;double best=INFINITY;
      for(int j=0;j<count;j++) {
       if(E(j,SIDE)==E(a,SIDE)||E(j,HP)<=0||(E(j,AIR)?!E(a,AIR_TARGET):!E(a,GROUND_TARGET)))continue;
       bool used=false;for(int v=0;v<nv;v++)if(victims[v]==j)used=true;if(used)continue;
       double gap=distance(e,previous,j);if(gap>E(a,CHAIN_RADIUS))continue;
       if(gap<best ||(gap==best && chosen>=0 && E(j,UID)<E(chosen,UID))){chosen=j;best=gap;}
      }
      if(chosen<0)break;victims[nv++]=chosen;
     }
     m[10]=(int)m[10]|1;
    } else if(E(a,ATTACK_TARGETS)>1) {
     while(nv<(int)E(a,ATTACK_TARGETS)) {
      int chosen=-1;double best=INFINITY;
      for(int j=0;j<count;j++) {
       if(E(j,SIDE)==E(a,SIDE)||E(j,HP)<=0||(E(j,AIR)?!E(a,AIR_TARGET):!E(a,GROUND_TARGET)))continue;
       bool used=false;for(int v=0;v<nv;v++)if(victims[v]==j)used=true;if(used)continue;
       double gap=distance(e,a,j);if(gap>E(a,REACH))continue;
       if(gap<best ||(gap==best && chosen>=0 && E(j,UID)<E(chosen,UID))){chosen=j;best=gap;}
      }
      if(chosen<0)break;victims[nv++]=chosen;
     }
    }
    double chainDelay=0.;int previous=a,bolts=E(a,CHAIN_COUNT)>1?nv:(int)E(a,ATTACK_TARGETS);
    for(int v=0;v<bolts;v++) {
     if(pending>=NI){codes[k]=2;return;}int victim=victims[v%nv];
     double delay=E(a,PROJECTILE_SPEED)?distance(e,a,victim)/E(a,PROJECTILE_SPEED):0.;
     if(E(a,CHAIN_COUNT)>1){chainDelay+=E(a,PROJECTILE_SPEED)?distance(e,previous,victim)/E(a,PROJECTILE_SPEED):0.;delay=chainDelay;previous=victim;}
     double* p=events+(pending++)*13;
     p[0]=time+delay;p[1]=E(a,SIDE);p[2]=E(victim,UID);
     p[3]=E(victim,X);p[4]=E(victim,Y);p[5]=damage;p[6]=E(a,SPLASH);p[7]=E(a,AIR_TARGET);p[8]=1.;
     p[9]=E(a,STUN);p[10]=E(a,PUSHBACK);p[11]=E(a,UID);p[12]=0;
    }
    double factor=(time<E(a,HASTE_UNTIL)?E(a,HASTE):1.)*(time<E(a,SLOW_UNTIL)?E(a,ATTACK_SLOW):1.);
    E(a,READY_AT)=time+E(a,PERIOD)/factor;if(E(a,TOWER))m[E(a,SIDE)==1?7:8]++;
   } else if(E(a,SPEED)>0 && !E(a,TOWER)) {
    E(a,WINDING_TARGET)=-1;double tx=E(t,X),ty=E(t,Y);
    if(grid) {if(!waypoint(e,count,a,t,b0,b1,tx,ty,cacheStart,cacheGoal,cacheHash,cacheNext,routeKeys,routeValues)) {
      if(cacheStart[a]==-2){codes[k]=3;return;}continue;
    }}
    else if(!E(a,AIR) && (E(a,Y)-16)*(ty-16)<0){tx=E(a,X)<9?b0:b1;ty=16-E(a,SIDE)*.5;}
    double length=hypot(tx-E(a,X),ty-E(a,Y));if(!length)continue;
    double multiplier=E(a,CHARGE_DISTANCE)&&E(a,WALKED)>=E(a,CHARGE_DISTANCE)?E(a,CHARGE_MULTIPLIER):1.;
    double factor=(time<E(a,HASTE_UNTIL)?E(a,HASTE):1.)*(time<E(a,SLOW_UNTIL)?E(a,SLOW):1.);
    double walk=fmin(E(a,SPEED)*multiplier*factor*dt,length);
    double nx=E(a,X)+(tx-E(a,X))/length*walk,ny=E(a,Y)+(ty-E(a,Y))/length*walk;
    if(grid)for(int j=0;j<count;j++)if(j!=a&&j!=t&&E(j,HP)>0&&E(j,AIR)==E(a,AIR)) {
     double r=fmin(.55,E(a,RADIUS)+E(j,RADIUS));double old=hypot(E(a,X)-E(j,X),E(a,Y)-E(j,Y)),n=hypot(nx-E(j,X),ny-E(j,Y));
     if(n<r && n<old){walk*=.5;nx=E(a,X)+(tx-E(a,X))/length*walk;ny=E(a,Y)+(ty-E(a,Y))/length*walk;}
    }
    E(a,WALKED)+=walk;E(a,X)=nx;E(a,Y)=ny;
   }
  }
  c.count=count;c.pending=pending;int before=count;
  deaths(c);if(*c.code)return;count=c.count;pending=c.pending;
  if(count!=before)for(int j=0;j<NE;j++)cacheStart[j]=-1;
 }
 m[2]=pending;
}
