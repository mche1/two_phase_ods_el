clear;
N=8000; %phase 1 sample size
beta0=[-4 1 1]';
alpha0=[-3.5,2.3,0,0]';
vars=[];
for(i=1:1000)
    rng(i) %random seed
    tic
    rho=0.1;
    mu=[0 0];
    sigma=[1 rho;rho 1];
    EXPL=mvnrnd(mu,sigma,N); % original X_tilde
    z=EXPL(:,1);
    x1=(EXPL(:,2)<.44&EXPL(:,2)>-0.44)+2*(EXPL(:,2)>.44);
    
    fulldesmat=[ones(N,1),z,x1];%full design matrix
    X=fulldesmat(:,1:length(beta0));
    muy=X*beta0;
    
    py=expit(muy);
    y=binornd(1,py);
    d=[.5,1.5];
    s1=(x1>d(1)&x1<d(2));
    s2=(x1>d(2));%%%define the 3 strata

    pr=expit([ones(N,1),y,s1,s2]*alpha0); %%%%assign 2 different sampling prob for the 3 strata
     r=binornd(1,pr);
     pop=[y,z.*r,x1,x1.*z.*r,s1,s2,r];
     
     mdpi1=fitglm([y],r,'Distribution','binomial'); %first working model of pi, using only y as predictor
     alpha1=table2array(mdpi1.Coefficients(:,1));

     mdpi2=fitglm([y,s1,s2],r,'Distribution','binomial'); % 2nd working model of pi, using y and s
     alpha2=table2array(mdpi2.Coefficients(:,1));
     w0=log(expit([ones(N,2),s1,s2]*alpha0)./expit([ones(N,1),zeros(N,1),s1,s2]*alpha0));
     w1=log(expit([ones(N,2)]*alpha1)./expit([ones(N,1),zeros(N,1)]*alpha1));
     w2=log(expit([ones(N,2),s1,s2]*alpha2)./expit([ones(N,1),zeros(N,1),s1,s2]*alpha2));
     mdcc=fitglm([z((r==1)),x1(r==1)],y(r==1),'Distribution','binomial');
     
     md0=fitglm([z((r==1)),x1(r==1)],y(r==1),'Distribution','binomial','Offset',w0(r==1));
     md1=fitglm([z((r==1)),x1(r==1)],y(r==1),'Distribution','binomial','Offset',w1(r==1));
     md2=fitglm([z((r==1)),x1(r==1)],y(r==1),'Distribution','binomial','Offset',w2(r==1));
     beta_cml0=table2array(md0.Coefficients(:,1));
      beta_cml0=fsolve(@(x)scml0(pop,x,alpha0),table2array(mdcc.Coefficients(:,1)));
      lb=length(beta_cml0);
     
      beta_cml1=table2array(md1.Coefficients(:,1));
      
      beta_cml2=table2array(md2.Coefficients(:,1));
     
     mdh=fitglm([x1],y,'Distribution','binomial');
     theta=table2array(mdh.Coefficients(:,1));
     para_sw1=fsolve(@(x)ssw(pop,x,lb),[beta_cml1;alpha1]);
     beta_sw1=para_sw1(1:lb);
     alpha_sw1=para_sw1((1+lb):end);
     options = optimoptions(@fsolve,'Algorithm','levenberg-marquardt','SpecifyObjectiveGradient',true);
     para_sw2=fsolve(@(x)ssw(pop,x,lb),[beta_cml2;alpha2],options);
     beta_sw2=para_sw2(1:lb);
     alpha_sw2=para_sw2((1+lb):end);
         
     la=length(alpha1);
     la2=length(alpha2);
     
     para_el1=el7(pop,beta_sw2,alpha1,theta); %% EL estimator with non-stratified pi model/alpha1
     beta_el1=para_el1(1:lb);
     alpha_el1=para_el1((lb+1):(lb+la));
     v_el1=v_el7(pop,beta_el1,alpha_el1,theta); %% variance estimate
     
     para_el2=el7(pop,beta_sw2,alpha2,theta);%% EL estimator with stratified pi model/alpha2
     beta_el2=para_el2(1:lb);
     alpha_el2=para_el2((lb+1):(lb+la2));
     v_el2=v_el7(pop,beta_el2,alpha_el2,theta);
   
     [beta_zlm v]=fle3_zlm_em(pop,beta_cml2); %%%% EM-based full ML estimator by Zhao, Lawless and McLeish
     parami=[i;beta_cml0;beta_cml1;beta_cml2;beta_sw1;beta_sw2;beta_el1;beta_el2;beta_zlm];
     sei=real([i;v_el1;v_el2]);

name1=strcat('beta_logit_exp_8k_',num2str(i),'.csv');
name2=strcat('se_logit_exp_8k_',num2str(i),'.csv');
csvwrite(name1,parami)
csvwrite(name2,sei)
toc
end
