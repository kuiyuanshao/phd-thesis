suppressMessages(lapply(c("ggplot2","dplyr","ggh4x"), require, character.only=TRUE))
ne <- new.env(); load("./simulations/results/results_COMBINED.RData", envir=ne)
SBP <- ne$combined_resultCoeff_SBP; HTN <- ne$combined_resultCoeff_HTN
OUT <- "./simulations/results"; dir.create(OUT, recursive=TRUE, showWarnings=FALSE)

fig_methods <- c("CC","MICE-PMM","MICE-RF","MIXGB","TABCSDI","GAIN","RAKING","SICG","SIRD")
method_disp <- c(CC="CC","MICE-PMM"="MICE-PMM","MICE-RF"="MICE-RF",MIXGB="MIXGB",TABCSDI="TabCSDI",GAIN="GAIN",RAKING="RAKING",SICG="SICG",SIRD="SIRD")
mlev <- unname(method_disp[fig_methods]); face_vec <- ifelse(mlev %in% c("SICG","SIRD"),"bold","plain")
design_disp <- c(SRS="SRS",RS="RS",WRS="WRS",ODS_TAIL="ODS-Tail",NEYMAN_ODS="Neyman-ODS",SFS="SFS",NEYMAN_INF="Neyman-IF")
dlev <- c("SRS","RS","WRS","ODS-Tail","Neyman-ODS","SFS","Neyman-IF")
design_cols <- c("SRS"="#999999","RS"="#A6CEE3","WRS"="#1F78B4","ODS-Tail"="#B2DF8A","Neyman-ODS"="#33A02C","SFS"="#FDBF6F","Neyman-IF"="#FF7F00")
focal_covs  <- c("ln_na_true","ln_na_true:c_age"); phase1_covs <- c("c_age","c_bmi","high_chol","usborn","female","bkg_pr","bkg_o")
cov_disp <- c("ln_na_true"="Exposure","ln_na_true:c_age"="Interaction","c_age"="Age","c_bmi"="BMI",
              "high_chol"="High chol.","usborn"="US-born","female"="Female","bkg_pr"="Bkg PR","bkg_o"="Bkg O")
focal_lev <- unname(cov_disp[focal_covs]); phase1_lev <- unname(cov_disp[phase1_covs])
tsbp <- c("ln_na_true"=27.4725275,"ln_na_true:c_age"=0.15,c_age=0.4060319,c_bmi=0.1703107,high_chol=1.6108133,usborn=-0.5636760,female=-8.9318846,bkg_pr=-1.6697425,bkg_o=-2.3219322)
thtn <- c("ln_na_true"=1.43901944,"ln_na_true:c_age"=0.005,c_age=0.08715117,c_bmi=0.05645171,high_chol=0.37938242,usborn=-0.21958482,female=-0.11976800,bkg_pr=-0.17504836,bkg_o=-0.84484199)
LAD <- c(20,30,40,50,75,100,150,200,300,500,1000,2000,5000,10000,50000); nicec <- function(x){ v<-LAD[which(LAD>=x)[1]]; if(is.na(v)) tail(LAD,1) else v }
allcovs <- c(focal_covs, phase1_covs)

build <- function(cf, model, truths){
  M <- as.character(cf$Method); Dg <- toupper(as.character(cf$Design)); ok <- M %in% fig_methods & Dg %in% names(design_disp)
  do.call(rbind, lapply(allcovs, function(cov){ est <- suppressWarnings(as.numeric(cf[[cov]]))
    data.frame(model=model, cov=unname(cov_disp[cov]), method=unname(method_disp[M[ok]]), design=unname(design_disp[Dg[ok]]), b=(est[ok]-truths[[cov]])/truths[[cov]]*100) }))
}
L <- rbind(build(SBP,"Gaussian",tsbp), build(HTN,"Binomial",thtn)) %>% filter(is.finite(b))
L$method <- factor(L$method, mlev); L$design <- factor(L$design, dlev); L$model <- factor(L$model, c("Gaussian","Binomial"))
bstat <- function(x){ x<-x[is.finite(x)]; if(length(x)<2) rep(NA_real_,5) else boxplot.stats(x)$stats }
clamp_stats <- function(dd, grpvars){
  gs <- dd %>% group_by(across(all_of(grpvars))) %>% summarise(s=list(bstat(b)), .groups="drop") %>%
    mutate(ymin=sapply(s,`[`,1),lower=sapply(s,`[`,2),mid=sapply(s,`[`,3),upper=sapply(s,`[`,4),ymax=sapply(s,`[`,5)) %>% select(-s) %>% filter(!is.na(mid))
  cv <- setdiff(grpvars, c("method","design"))
  cl <- gs %>% group_by(across(all_of(cv))) %>% summarise(clip=nicec(quantile(abs(c(lower,upper)),0.90,na.rm=TRUE)), .groups="drop")
  gs %>% left_join(cl, by=cv) %>% mutate(across(c(ymin,lower,mid,upper,ymax), ~pmax(-clip,pmin(clip,.))))
}
boxlayer <- function() list(
  geom_hline(yintercept=0, linetype="dashed", colour="grey25", linewidth=0.4),
  geom_boxplot(aes(ymin=ymin,lower=lower,middle=mid,upper=upper,ymax=ymax), stat="identity", fill=NA, linewidth=0.34, width=0.62, position=position_dodge2(padding=0.12)),
  geom_blank(aes(y=clip)), geom_blank(aes(y=-clip)),
  scale_colour_manual(values=design_cols, name="Sampling Design"), labs(x="Method", y="Percentage Bias (%)"))
base_theme <- function() theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold",size=12), strip.background=element_rect(fill="grey92",colour="grey55",linewidth=0.4), strip.text=element_text(face="bold",size=9.5),
        panel.grid.minor=element_blank(), panel.grid.major.x=element_blank(), axis.text.x=element_text(angle=30,hjust=1,face=face_vec,size=8), legend.position="bottom")

gd <- clamp_stats(L %>% filter(cov %in% focal_lev) %>% mutate(cov=factor(cov, focal_lev)), c("cov","model","method","design"))
p <- ggplot(gd, aes(method, colour=design)) + boxlayer() + facet_grid2(cov ~ model, scales="free_y", independent="y") +
  labs(title="Simulation Design II — Exposure and Interaction") + base_theme() + guides(colour=guide_legend(nrow=1))
ggsave(file.path(OUT,"simI_focal_pctbias.pdf"), p, width=10, height=7); ggsave(file.path(OUT,"simI_focal_pctbias.png"), p, width=10, height=7, dpi=200)

save_other <- function(mo, base){
  gd <- clamp_stats(L %>% filter(model==mo, cov %in% phase1_lev) %>% mutate(cov=factor(cov, phase1_lev)), c("cov","method","design"))
  p <- ggplot(gd, aes(method, colour=design)) + boxlayer() + facet_wrap(~cov, ncol=3, scales="free_y") +
    labs(title=paste0("Simulation Design II — Phase-One Covariates (", mo, ")")) + base_theme() + guides(colour=guide_legend(nrow=1))
  ggsave(file.path(OUT,paste0(base,".pdf")), p, width=15, height=9); ggsave(file.path(OUT,paste0(base,".png")), p, width=15, height=9, dpi=200)
}
save_other("Gaussian", "simI_phase1cov_gaussian_pctbias")
save_other("Binomial", "simI_phase1cov_binomial_pctbias")
cat("done ->", OUT, "\n")
