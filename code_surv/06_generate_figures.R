suppressMessages(lapply(c("ggplot2","dplyr","ggh4x"), require, character.only=TRUE))
ne <- new.env(); load("./simulations/results/results_COMBINED.RData", envir=ne)
OUT <- "./simulations/results"; dir.create(OUT, recursive=TRUE, showWarnings=FALSE)

fig_methods <- c("CC","MICE-PMM","MICE-RF","MIXGB","GAIN","RAKING","SICG","SIRD")
method_disp <- setNames(fig_methods, fig_methods); mlev <- fig_methods; face_vec <- ifelse(mlev %in% c("SICG","SIRD"),"bold","plain")
design_disp <- c(SRS="SRS",BALANCE="Balanced",NEYMAN="Neyman-IF"); dlev <- c("SRS","Balanced","Neyman-IF")
design_cols <- c("SRS"="#999999","Balanced"="#1F78B4","Neyman-IF"="#FF7F00")
scen_lev <- c("SampleE","SampleOE")
cont_imp <- c("HbA1c","eGFR","Insulin","BMI")
cat_imp  <- c("rs4506565 1","rs4506565 2","Smoke 2","Smoke 3")
inter_cov<- "Age:Insulin"
phase1   <- c("Age","Sex TRUE","Insurance TRUE","Race AFR","Race AMR","Race SAS","Race EAS")
cov_disp <- c("HbA1c"="HbA1c","eGFR"="eGFR","Insulin"="Insulin","BMI"="BMI",
              "rs4506565 1"="rs4506565 (1)","rs4506565 2"="rs4506565 (2)","Smoke 2"="Smoking (2)","Smoke 3"="Smoking (3)",
              "Age:Insulin"="Interaction",
              "Age"="Age","Sex TRUE"="Sex","Insurance TRUE"="Insurance","Race AFR"="Race AFR","Race AMR"="Race AMR","Race SAS"="Race SAS","Race EAS"="Race EAS")
LAD <- c(5,10,15,20,25,30,40,50,75,100,150,200,300,500,1000); nicec <- function(x){ v<-LAD[which(LAD>=x)[1]]; if(is.na(v)) tail(LAD,1) else v }

build_long <- function(covset){
  rows <- list(); k <- 0
  for(sc in scen_lev){ cf <- ne[[paste0("combined_resultCoeff_",sc)]]
    M <- as.character(cf$Method); Dg <- toupper(as.character(cf$Design)); ok <- M %in% fig_methods & Dg %in% names(design_disp)
    for(cov in covset){ est <- suppressWarnings(as.numeric(cf[[cov]])); tv <- mean(est[M=="VAL"], na.rm=TRUE)
      k <- k+1; rows[[k]] <- data.frame(scenario=sc, cov=unname(cov_disp[cov]), method=unname(method_disp[M[ok]]), design=unname(design_disp[Dg[ok]]), b=(est[ok]-tv)/tv*100) } }
  d <- do.call(rbind, rows) %>% filter(is.finite(b))
  d$scenario<-factor(d$scenario,scen_lev); d$method<-factor(d$method,mlev); d$design<-factor(d$design,dlev); d
}
bstat <- function(x){ x<-x[is.finite(x)]; if(length(x)<2) rep(NA_real_,5) else boxplot.stats(x)$stats }
clamp <- function(dd, facetvars){
  gs <- dd %>% group_by(across(all_of(c(facetvars,"method","design")))) %>% summarise(s=list(bstat(b)), .groups="drop") %>%
    mutate(ymin=sapply(s,`[`,1),lower=sapply(s,`[`,2),mid=sapply(s,`[`,3),upper=sapply(s,`[`,4),ymax=sapply(s,`[`,5)) %>% select(-s) %>% filter(!is.na(mid))
  cl <- gs %>% group_by(across(all_of(facetvars))) %>% summarise(clip=nicec(quantile(abs(c(lower,upper)),0.90,na.rm=TRUE)), .groups="drop")
  gs %>% left_join(cl, by=facetvars) %>% mutate(across(c(ymin,lower,mid,upper,ymax), ~pmax(-clip,pmin(clip,.))))
}
boxlayer <- list(
  geom_hline(yintercept=0, linetype="dashed", colour="grey25", linewidth=0.4),
  geom_boxplot(aes(ymin=ymin,lower=lower,middle=mid,upper=upper,ymax=ymax), stat="identity", fill=NA, linewidth=0.34, width=0.66, position=position_dodge2(padding=0.16)),
  geom_blank(aes(y=clip)), geom_blank(aes(y=-clip)),
  scale_colour_manual(values=design_cols, name="Sampling Design"), labs(x="Method", y="Percentage Bias (%)"))
base_theme <- theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold",size=12), strip.background=element_rect(fill="grey92",colour="grey55",linewidth=0.4), strip.text=element_text(face="bold",size=9.5),
        panel.grid.minor=element_blank(), panel.grid.major.x=element_blank(), axis.text.x=element_text(angle=25,hjust=1,face=face_vec,size=8.5), legend.position="bottom")

save_grid <- function(covset, base, ttl, ht){
  gd <- clamp(build_long(covset) %>% mutate(cov=factor(cov, unname(cov_disp[covset]))), c("scenario","cov"))
  p <- ggplot(gd, aes(method, colour=design)) + boxlayer + facet_grid2(cov ~ scenario, scales="free_y", independent="y") +
    labs(title=ttl) + base_theme + guides(colour=guide_legend(nrow=1))
  ggsave(file.path(OUT,paste0(base,".pdf")), p, width=9.5, height=ht); ggsave(file.path(OUT,paste0(base,".png")), p, width=9.5, height=ht, dpi=200)
}
save_interaction <- function(base, ttl){
  gd <- clamp(build_long(inter_cov), c("scenario"))
  p <- ggplot(gd, aes(method, colour=design)) + boxlayer + facet_wrap(~scenario) +
    labs(title=ttl) + base_theme + guides(colour=guide_legend(nrow=1))
  ggsave(file.path(OUT,paste0(base,".pdf")), p, width=10, height=5); ggsave(file.path(OUT,paste0(base,".png")), p, width=10, height=5, dpi=200)
}
save_grid(cont_imp, "surv_continuous_phase2_pctbias", "Simulation Design III — Continuous Phase-Two Covariates", 10)
save_grid(cat_imp,  "surv_categorical_phase2_pctbias", "Simulation Design III — Binary & Categorical Phase-Two Covariates", 10)
save_interaction("surv_interaction_pctbias", "Simulation Design III — Interaction")
save_grid(phase1,   "surv_phase1cov_pctbias", "Simulation Design III — Phase-One Covariates", 15)
cat("done ->", OUT, "\n")
