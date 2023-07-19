 function u=ufunc_log(pop,para,alpha,theta)
 siz=size(pop);
 lb=length(para);
 la=length(alpha);
 n=siz(1);
 d=length(theta);
 u=zeros(n,d);
 y=pop(:,1);
 z=pop(:,2);
 x=pop(:,3);
 s1=pop(:,lb+3);
 s2=pop(:,lb+4);
 cov=[ones(n,1),z,x];
 picov=[ones(n,1),y,s2,y.*s2];
 picov0=[ones(n,1),zeros(n,1),s2,zeros(n,1)];
 picov1=[ones(n,2),s2,s2];
 picov=picov(:,1:la);
 picov0=picov0(:,1:la);
 picov1=picov1(:,1:la);
 muy=cov*para(1:end);
 numerator=zeros(n,2);
 denom=zeros(n,1);
 for(i=1:n)
     if(pop(i,end)==1)
      %  numerator(i,1)=quadgk(@(y)(y-theta(1)-pop(i,2)*theta(2)).*exp(-(y-muy(i)).^2/(2*s.^2)),-100,100);
      numerator(i,1:2)=hfunc_log(1,x(i,:),theta)*expit(cov(i,:)*para)+hfunc_log(0,x(i,:),theta)*(1-expit(cov(i,:)*para)); 
      %numerator(i,2)=quadgk(@(y)(y-theta(1)-pop(i,3)*theta(2))*pop(i,3).*normpdf(y,muy(i),s),-100,100);
       % numerator(i,3)=quadgk(@(y)((y-theta(1)-pop(i,2)*theta(2))^2/theta(3)^3-1/theta(3)).*exp(-(y-muy(i)).^2/(2*s.^2)),-100,100)
        denom(i)=expit(cov(i,:)*para)*expit(picov1(i,:)*alpha)+(1-expit(cov(i,:)*para))*expit(picov0(i,:)*alpha);
     else
        numerator(i,:)=0;
        denom(i)=1;
     end
         
 end
 den=repmat(denom,[1,d]);
 u=numerator./den;
end