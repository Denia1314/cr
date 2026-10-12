// Complete reference mechanisms. Each context belongs to one branch/thread.
struct CombatContext {
 double *e,*impacts,*fx,*scratch,*meta,*prototypes,*pulses;
 unsigned char* spawned;int* code;int count,pending,effects,entityCapacity,effectCapacity;
};
__device__ int by_uid(CombatContext &c,double uid) {
 for(int i=0;i<c.count;i++)if(c.e[i*NF+UID]==uid)return i;return -1;
}
__device__ bool damage_base(CombatContext &c,int i,double damage) {
 double* e=c.e;
 if(E(i,HP)<=0 || (E(i,DASH_IMMUNE) && E(i,DASH_PHASE)==2))return false;
 if(E(i,SHIELD)>0)E(i,SHIELD)=fmax(0.,E(i,SHIELD)-damage);
 else {double dealt=fmin(E(i,HP),fmax(0.,damage));E(i,HP)-=dealt;if(E(i,TOWER))c.meta[E(i,SIDE)==1?5:6]+=dealt;}
 return true;
}
__device__ bool hit(CombatContext &c,int i,double damage,int attacker=-1) {
 double* e=c.e;
 if(E(i,HP)<=0 || (E(i,DASH_IMMUNE) && E(i,DASH_PHASE)==2))return false;
 if(attacker>=0 && E(i,REFLECTED_DAMAGE) && E(attacker,SIDE)!=E(i,SIDE) && distance(e,i,attacker)<=E(i,REFLECTED_RADIUS)) {
  if(damage_base(c,attacker,E(i,REFLECTED_DAMAGE)))control(e,attacker,c.meta[0],E(i,REFLECTED_STUN),0.,0.,0.);
  c.meta[10]=(int)c.meta[10]|4;
 }
 return damage_base(c,i,damage);
}
__device__ bool append_impact(CombatContext &c,double at,double side,double target,double x,double y,
 double damage,double radius,double air,double tower,double stun=0.,double push=0.,double attacker=0.) {
 if(c.pending>=NI){*c.code=2;return false;}
 double* p=c.impacts+(c.pending++)*13;
 double values[13]={at,side,target,x,y,damage,radius,air,tower,stun,push,attacker,0.};
 for(int f=0;f<13;f++)p[f]=values[f];return true;
}
__device__ bool append_effect(CombatContext &c,const double* source,double* dest,int &size) {
 if(size>=c.effectCapacity){*c.code=4;return false;}
 for(int f=0;f<EF;f++)dest[size*EF+f]=source[f];size++;return true;
}
__device__ int spawn(CombatContext &c,int spec,double side,double x,double y,bool deployed,
 bool cloned=false,double value=0.,bool copyShield=true,bool clonedSpecShield=true,double* output=nullptr,int* size=nullptr) {
 if(spec<0)return -1;
 if(c.count>=c.entityCapacity){*c.code=2;return -1;}
 double* e=c.e;int i=c.count++;double time=c.meta[0];
 for(int f=0;f<NF;f++)E(i,f)=c.prototypes[spec*NF+f];
 E(i,UID)=c.meta[11]++;E(i,SIDE)=side;E(i,X)=x;E(i,Y)=y;E(i,VALUE)=value;E(i,CLONED)=cloned;
 E(i,HP)=cloned?1.:E(i,SPEC_HP);if(cloned)E(i,SHIELD)=clonedSpecShield?(E(i,SHIELD)>0?1.:0.):(copyShield?1.:0.);
 E(i,READY_AT)=E(i,ACTIVE_AT)=time+(deployed?E(i,DEPLOY):0.);
 E(i,EXPIRES_AT)=E(i,LIFETIME)?time+E(i,LIFETIME):INFINITY;
 E(i,SPAWN_AT)=E(i,HAS_SPAWN)?time+E(i,SPAWN_START):INFINITY;
 E(i,RESOURCE_AT)=E(i,RESOURCE_PERIOD)?time+E(i,DEPLOY)+E(i,RESOURCE_PERIOD):INFINITY;
 c.spawned[spec]=1;
 int pulse=(int)E(i,PULSE_TEMPLATE);
 if(deployed && pulse>=0) {
  double event[EF];for(int f=0;f<EF;f++)event[f]=c.pulses[pulse*EF+f];
  event[FX_AT]=E(i,ACTIVE_AT);event[FX_SIDE]=side;event[FX_X]=x;event[FX_Y]=y;event[FX_OWNER]=E(i,UID);
  if(output)append_effect(c,event,output,*size);else append_effect(c,event,c.fx,c.effects);
 }
 return i;
}
__device__ void pulse_effect(CombatContext &c,const double* p) {
 double* e=c.e;
 if(p[FX_OWNER]>=0) {int owner=by_uid(c,p[FX_OWNER]);if(owner<0 || E(owner,HP)<=0)return;}
 for(int i=0;i<c.count;i++) {
  if(E(i,SIDE)==p[FX_SIDE]||E(i,HP)<=0||(!p[FX_AIR]&&E(i,AIR))||(!p[FX_GROUND]&&!E(i,AIR)))continue;
  if(hypot(E(i,X)-p[FX_X],E(i,Y)-p[FX_Y])>p[FX_RADIUS]+E(i,RADIUS))continue;
  if(!hit(c,i,p[FX_DAMAGE]*(E(i,TOWER)?p[FX_TOWER_MULTIPLIER]:1.)))continue;
  control(e,i,c.meta[0],p[FX_STUN],p[FX_PUSHBACK],p[FX_X],p[FX_Y]);
  if(p[FX_DURATION] &&(p[FX_SLOW]<1 ||p[FX_ATTACK_SLOW]<1)) {
   E(i,SLOW_UNTIL)=fmax(E(i,SLOW_UNTIL),c.meta[0]+p[FX_DURATION]);
   E(i,SLOW)=fmax(.1,p[FX_SLOW]);E(i,ATTACK_SLOW)=fmax(.1,p[FX_ATTACK_SLOW]);
  }
 }
}
__device__ void effects(CombatContext &c) {
 double* e=c.e;int old=c.effects,out=0;double time=c.meta[0];
 for(int j=0;j<old;j++) {
  double p[EF];for(int f=0;f<EF;f++)p[f]=c.fx[j*EF+f];
  if(p[FX_AT]>time){append_effect(c,p,c.scratch,out);continue;}
  int pattern=(int)p[FX_PATTERN];
  if(pattern==PAT_UNIT_PULSE){pulse_effect(c,p);continue;}
  if(pattern==PAT_SPAWN_ENTITY) {
   if(c.count<96)spawn(c,(int)p[FX_SPAWN_SPEC],p[FX_SIDE],p[FX_X],p[FX_Y],true,false,p[FX_SUMMON_VALUE],true,true,c.scratch,&out);
   continue;
  }
  int targets[NE],nt=0;
  for(int i=0;i<c.count;i++)if(E(i,HP)>0 &&(p[FX_FRIENDLY]?E(i,SIDE)==p[FX_SIDE]:E(i,SIDE)!=p[FX_SIDE]) &&(p[FX_AIR]||!E(i,AIR))) {
   if(pattern!=PAT_LINE && pattern!=PAT_LINE_SPAWN && hypot(E(i,X)-p[FX_X],E(i,Y)-p[FX_Y])>p[FX_RADIUS]+E(i,RADIUS))continue;
   targets[nt++]=i;
  }
  if(pattern==PAT_HIGHEST_HP) {
   for(int i=1;i<nt;i++) {
    int next=targets[i],at=i;
    while(at>0 && (E(next,HP)>E(targets[at-1],HP)||(E(next,HP)==E(targets[at-1],HP)&&E(next,UID)<E(targets[at-1],UID)))){targets[at]=targets[at-1];at--;}
    targets[at]=next;
   }
   nt=min(nt,(int)p[FX_PULSES]);
  }
  for(int v=0;v<nt;v++) {
   int i=targets[v];
   if(pattern==PAT_LINE || pattern==PAT_LINE_SPAWN) {
    double forward=(p[FX_Y]-E(i,Y))*p[FX_SIDE];
    if(-E(i,RADIUS)<=forward && forward<=p[FX_LENGTH]+E(i,RADIUS)&&fabs(E(i,X)-p[FX_X])<=p[FX_WIDTH]/2+E(i,RADIUS))
     append_impact(c,time+fmax(0.,forward)/fmax(.1,p[FX_SPEED]),p[FX_SIDE],E(i,UID),E(i,X),E(i,Y),
       p[FX_DAMAGE]*(E(i,TOWER)?p[FX_TOWER_MULTIPLIER]:1.),0.,false,1.,0.,p[FX_PUSHBACK]);
    continue;
   }
   if(pattern==PAT_CLONE) {
    if(!E(i,TOWER)&&!E(i,BUILDING)&&!E(i,CLONED)&&c.count<96)
     spawn(c,(int)E(i,PROTOTYPE),p[FX_SIDE],E(i,X)+.3,E(i,Y),false,true,0.,E(i,SHIELD)>0,false,c.scratch,&out);
    continue;
   }
   if(pattern==PAT_RAGE && E(i,SIDE)==p[FX_SIDE]) {
    E(i,HASTE_UNTIL)=fmax(E(i,HASTE_UNTIL),time+p[FX_DURATION]);E(i,HASTE)=1+p[FX_BOOST];continue;
   }
   double mult=E(i,TOWER)?p[FX_TOWER_MULTIPLIER]:(E(i,BUILDING)?p[FX_BUILDING_MULTIPLIER]:1.);
   if(!hit(c,i,p[FX_DAMAGE]*mult))continue;
   if(p[FX_STUN])control(e,i,time,p[FX_STUN],0.,0.,0.);
   if(!E(i,TOWER)&&!E(i,BUILDING)) {
    if(p[FX_SLOW]){E(i,SLOW_UNTIL)=fmax(E(i,SLOW_UNTIL),time+p[FX_PERIOD]);E(i,SLOW)=fmax(.1,1+p[FX_SLOW]);}
    if(pattern==PAT_PULL){E(i,X)+=(p[FX_X]-E(i,X))*.25;E(i,Y)+=(p[FX_Y]-E(i,Y))*.25;}
   }
  }
  if(p[FX_SPAWN_SPEC]>=0) {
   double y=p[FX_Y];if(pattern==PAT_LINE_SPAWN)y-=p[FX_SIDE]*p[FX_LENGTH];
   for(int i=0;i<min(20,(int)p[FX_SPAWN_COUNT]);i++) {
    double child[EF];for(int f=0;f<EF;f++)child[f]=p[f];child[FX_PATTERN]=PAT_SPAWN_ENTITY;
    double delay=pattern==PAT_SERIAL_SPAWN?i*p[FX_SPAWN_PERIOD]:(pattern==PAT_LINE_SPAWN?p[FX_LENGTH]/fmax(.1,p[FX_SPEED]):0.);
    child[FX_AT]=time+delay;child[FX_X]=p[FX_X]+((i%3)-1)*.4;child[FX_Y]=y+(i/3)*.3;
    append_effect(c,child,c.scratch,out);
   }
  }
  if(pattern==PAT_RAGE)for(int i=0;i<c.count;i++)if(E(i,SIDE)!=p[FX_SIDE] && hypot(E(i,X)-p[FX_X],E(i,Y)-p[FX_Y])<=p[FX_RADIUS]+E(i,RADIUS))
   hit(c,i,p[FX_DAMAGE]*(E(i,TOWER)?p[FX_TOWER_MULTIPLIER]:1.));
  if(*c.code)return;
 }
 for(int i=0;i<out*EF;i++)c.fx[i]=c.scratch[i];c.effects=out;
}
__device__ bool dash(CombatContext &c,int i,int target) {
 double* e=c.e;double time=c.meta[0];if(!E(i,DASH_DAMAGE))return false;
 if(E(i,DASH_PHASE)==1) {
  int victim=by_uid(c,E(i,DASH_TARGET));
  if(victim<0 ||E(victim,HP)<=0 || distance(e,i,victim)>E(i,DASH_MAX)){E(i,DASH_PHASE)=0;return false;}
  if(time<E(i,DASH_AT))return true;
  E(i,DASH_X)=E(victim,X);E(i,DASH_Y)=E(victim,Y);
  double travel=E(i,DASH_TRAVEL)?E(i,DASH_TRAVEL):hypot(E(victim,X)-E(i,X),E(victim,Y)-E(i,Y))/fmax(.1,E(i,DASH_SPEED));
  E(i,DASH_AT)=time+travel;E(i,DASH_PHASE)=2;return true;
 }
 if(E(i,DASH_PHASE)==2) {
  if(time<E(i,DASH_AT))return true;
  E(i,X)=E(i,DASH_X);E(i,Y)=E(i,DASH_Y);E(i,DASH_PHASE)=0;E(i,WALKED)=0;
  E(i,READY_AT)=time+E(i,DASH_RECOVERY)+E(i,FIRST_HIT);E(i,DASH_READY_AT)=E(i,READY_AT);E(i,WINDING_TARGET)=-1;
  append_impact(c,time,E(i,SIDE),E(i,DASH_TARGET),E(i,X),E(i,Y),E(i,DASH_DAMAGE),E(i,DASH_RADIUS),E(i,AIR_TARGET),1.,0.,E(i,DASH_PUSHBACK),E(i,UID));return true;
 }
 if(target>=0 && time>=fmax(E(i,READY_AT),E(i,DASH_READY_AT))) {
  double d=distance(e,i,target);
  if(E(i,DASH_MIN)<=d && d<=E(i,DASH_MAX)) {
   E(i,DASH_PHASE)=1;E(i,DASH_TARGET)=E(target,UID);E(i,DASH_AT)=time+E(i,DASH_WINDUP);E(i,WINDING_TARGET)=-1;
   c.meta[10]=(int)c.meta[10]|2;return true;
  }
 }
 return false;
}
__device__ void deaths(CombatContext &c) {
 double* e=c.e;double dead[NE*13];int nd=0,survivors=0;
 for(int i=0;i<c.count;i++) {
  if(E(i,HP)>0) {
   if(survivors!=i)for(int f=0;f<NF;f++)E(survivors,f)=E(i,f);
   survivors++;continue;
  }
  double* r=dead+(nd++)*13;
  double values[13]={E(i,DEATH_SPEC),E(i,DEATH_COUNT),E(i,SIDE),E(i,X),E(i,Y),E(i,CLONED),
   E(i,RESOURCE_ON_DEATH),E(i,DEATH_DAMAGE),E(i,DEATH_RADIUS),E(i,DEATH_DELAY),E(i,DEATH_AIR),E(i,DEATH_TOWER_MULTIPLIER),0.};
  for(int f=0;f<13;f++)r[f]=values[f];
 }
 c.count=survivors;
 for(int i=0;i<nd;i++) {
  double* r=dead+i*13;
  if(r[6])c.meta[r[2]==1?3:4]=fmin(10.,c.meta[r[2]==1?3:4]+r[6]);
  if(r[7])append_impact(c,c.meta[0]+r[9],r[2],-1.,r[3],r[4],r[7],r[8],r[10],r[11]);
  if(r[0]>=0 && c.count<96)for(int j=0;j<min(8,(int)r[1]);j++)spawn(c,(int)r[0],r[2],r[3]+j*.35,r[4],true,r[5]);
 }
 if(c.count>=96)c.meta[10]=(int)c.meta[10]|8;
 c.meta[1]=c.count;c.meta[2]=c.pending;c.meta[12]=c.effects;
}
