function [param v]=fle3_zlm_em(pop,param)
 MaxIter=200;MaxTol=5e-5;
 S=size(pop);
 n=sum(pop(:,end));
 g=ones(n,3)/n; % the prob. g_i's, where the (r i)-th entry stands for g(z_r|x_i)
 for(iter=1:MaxIter)
     pweight=profilewt(pop,param,g);
     options = optimoptions('fsolve','SpecifyObjectiveGradient',true,'MaxFunctionEvaluations',200,'MaxIterations',100,'Display','off');
     paramnew=fsolve(@(x)scorep(pop,x,pweight),param,options);
     g=updateg(pop,pweight);
  if(max(abs(paramnew-param))<MaxTol)
     break;
 else
     param=paramnew;
 end
 end
 
 [score J]=scorep(pop,param,pweight);
 v=diag(-J);
end

function w=profilewt(pop,beta,g)  %%%%function to compute the profile weights
 %s=size(pop);
 Vind=find(pop(:,end)); %%% the indices of fully observed cases
 Vcind=find(1-pop(:,end));%%%% indices of partially observed cases
 m=length(Vind);  %%% number of possible z values
 K=2;%length(Vcind); %%% K is the number of values y can take for those not fully observed
 Z=pop(Vind,2);   %%% all the z values
 w=zeros(m,3,K);    %%%profile weight matrix, each stands for P(z|x,y), 1st index for z, 2nd for x, 3rd for y. 
%  w2=zeros(m,K);
%  w3=zeros(m,K);
 for(i=0:1)  % i is the index of Y 
     for(j=1:3) % j is the index of X
        muy=[ones(m,1),Z,repmat(j-1,[m,1])]*beta;
        numerator1=g(:,j).*(i.*expit(muy)+(1-i).*(1-expit(muy)));
%      numerator2=g(:,i).*(normcdf((c(2)-muy)/s)-normcdf((c(1)-muy)/s));
%      numerator3=g(:,i).*(1-normcdf((c(2)-muy)/s));
        denom1=sum(numerator1);
%         denom2=sum(numerator2);
%            denom3=sum(numerator3);
        w(:,j,(i+1))=numerator1/denom1;%%%cond. pmf of z given y=i, x=j-1
%      w3(:,i)=numerator3/denom3;
      end
%  w=cat(3,w1,w2,w3);
 %%%w is a 3-dim array, w(r,i,j)=P(X=xr|y in Sj, z=zi)
 end
end
function [score J]=scorep(pop,param,w) %%%% the score function of the parameter of interest
 sb=[];
 sw=size(w);
 sp=size(pop);
 N=sp(1);
 n=sw(1);
 m=n;
 Vind=find(pop(:,end));
 Vcind=find(1-pop(:,end));
 Y=pop(Vcind,1);
 Z=pop(Vind,2);
 beta=param(1:3);
 for(k=1:N)
     y=pop(k,1);
     x=pop(k,3);
     if(pop(k,end)==1) %%%% this is a fully observed case
         z=pop(k,2);
         muy=expit([1,z,x]*param(1:3));
         sb=[sb;((y-muy))*[1,z,x]];
     else         %%%%%%%%% this is a partially observed case
         cov=[ones(m,1),Z,repmat(x,[m,1])];
         muy=expit(cov*beta);
         weightedsumb=sum(repmat(w(:,(x+1),(y+1)).*(y-muy),[1,3]).*cov);
         sb=[sb;weightedsumb];
     end
 end
  score=sum([sb]);
  J=-[sb]'*[sb];
end
function g=updateg(pop,pweight)  %%%%%%%%%%%%% g(r,i)=g(z_r|x=i-1) 
 siz=size(pweight);
 n=siz(1);
 siz2=size(pop);
 N=siz2(1);
 g=zeros(n,3);
 Vind=find(pop(:,end));
 Vindc=find(1-pop(:,end));
 pop_bar=pop(Vindc,:);
%  Vindc1=find(pop(:,end)==0&pop(:,2)==0);
%  Vindc2=find(pop(:,end)==0&pop(:,2)==1);
%  Vindc3=find(pop(:,end)==0&pop(:,2)==2);
% V=[length(Vindc1),length(Vindc2),length(Vindc3)];%%%%
 Z=pop(Vind,2); %%% all the Z values observed 
 xz=pop(Vind,3); %%the X values with Z observed
 X=pop(:,3);
 V0=sum(X==0);
 V1=sum(X==1);
 V2=N-V0-V1;
 V=[V0,V1,V2];
 for(i=1:3) %the index for x; in fact x=i-1
     ind=find(pop_bar(:,3)==i-1);%%%%%%%%%% indices of those with unobserved z, x=i-1.
     for(r=1:n)%%the index for z
         z=Z(r);
         sum1=sum(pop(:,2)==z&pop(:,3)==i-1);
%       if(sum1==1) %%%this z value is observed together with this x value
%           g(r,i)=(1+sum(pweight(r,ind)))/V(i);
%           %g(r,i)=1+[sum(Y<c(1)&R==0&Z==i-1),sum(Y>c(1)&Y<c(2)&R==0&X==i-1),sum(Y>c(2)&R==0&X==i-1)]*[pweight(r,i,1),pweight(r,i,2),pweight(r,i,3)]';
%       else
         sum2=0;
         for(j=1:length(ind))
             %%%  ind(j) is the index of this case in pop_bar
             %%%% 
             sum2=sum2+pweight(r,pop_bar(ind(j),3)+1,pop_bar(ind(j),1)+1);
         end
          g(r,i)=(sum1+sum2)/V(i);
         %g(r,i)=[sum(Y<c(1)&R==0&Z==i-1),sum(Y>c(1)&Y<c(2)&R==0&X==i-1),sum(Y>c(2)&R==0&X==i-1)]*[pweight(r,i,1),pweight(r,i,2),pweight(r,i,3)]';
%       end
     end
 end
end