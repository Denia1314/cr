// Fuse the existing float64 spatial/DPS screen into one launch per roster member.
// One thread scores one point; enemy and tower reductions retain input order.
__device__ double positive(double v) { return fmax(0.,v); }
__device__ double pull_at(const double* e,double distance,double radius,bool building) {
    double pull=e[4];
    if(building) {
        bool viable=distance-radius-e[2]<=e[10] && distance-radius-e[2]<e[11];
        pull*=1-e[12]+e[12]*viable;
    }
    return pull;
}
extern "C" __global__ void rank_points(const double* points,const double* enemies,
    const double* towers,const double* eligible,double* out,int count,int ne,int nt,
    double radius,double reach,double speed,double damage,double period,double hp,
    double shield,double first_hit,int building,int trajectory,int ticks,double step) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    double x=points[2*i],y=points[2*i+1],incoming=0.,hit_count=0.;
    for(int j=0;j<ne;j++) {
        const double* e=enemies+16*j;
        double dx=x-e[0],dy=y-e[1],distance=sqrt(dx*dx+dy*dy);
        double pull=pull_at(e,distance,radius,building);
        double contact=positive(distance-reach-radius-e[2])/(.5+positive(speed+e[13]*pull-.5));
        incoming+=e[8]*pull*exp(-contact/2);
        hit_count+=e[3];
    }
    double lifetime=(hp+shield)/(incoming+.001),total=0.;
    for(int j=0;j<ne;j++) {
        const double* e=enemies+16*j;
        double dx=x-e[0],dy=y-e[1],distance=sqrt(dx*dx+dy*dy);
        double pull=pull_at(e,distance,radius,building);
        double contact=positive(distance-reach-radius-e[2])/(.5+positive(speed+e[13]*pull-.5));
        double fx=e[0]+dx*.5*pull,fy=e[1]+dy*.5*pull,tower_dps=0.;
        for(int k=0;k<nt;k++) {
            const double* t=towers+4*k;
            double tx=fx-t[0],ty=fy-t[1];
            if(sqrt(tx*tx+ty*ty)<=t[2]+e[2])tower_dps+=eligible[j*nt+k]*t[3];
        }
        double value=e[6]*exp(-contact/2)*((2+2*e[14])*e[3]+pull*(1+e[15])+tower_dps/100);
        value-=e[6]*fabs(distance-fmin(4.,reach)*.85)*.15*e[3];
        if(reach>2)value-=e[6]*positive(fmin(reach,4.)-distance)*.8*e[3]*pull;
        value-=e[6]*(1-e[3])*(1-pull);
        if(trajectory) {
            double own_dps=damage/fmax(.1,period)*e[3]/fmax(1.,hit_count);
            double shared_tower=tower_dps/fmax(1.,ne);
            double cap=positive(lifetime-contact-first_hit);
            double stall=positive(lifetime-contact)*pull*(contact<e[9]);
            double arrival=e[9]+stall,tower_damage=0.;
            for(int k=0;k<ticks;k++) {
                double time=k*step,active=fmin(positive(time-contact-first_hit),cap);
                double left=e[7]-own_dps*active-shared_tower*time;
                if(left>0 && time>=arrival)tower_damage+=e[8]*step;
            }
            value-=tower_damage/150;
        }
        total+=value;
    }
    out[i]=total;
}
