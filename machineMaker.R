# machineMaker.R
# Contains utility functions for machine learning projects.

#' Row-bind with fill
#'
#' Combines data frames or matrices by rows, filling missing columns with a specified value.
#' @param x A data frame, matrix, or list of these.
#' @param ... Additional data frames or matrices to combine.
#' @param fill Value to use for missing columns. Default is NULL.
#' @return A combined data frame or matrix.
rBind.fill<-function(x,...,fill=NULL)
{
  if (is.list(x) && !is.data.frame(x) && missing(...)) {
    Reduce(function (x,y) rBind.fill.internal(x,y,fill),x)
  }
  else {
    Reduce(function (x,y) rBind.fill.internal(x,y,fill),list(x,...))
  }
}

#' Internal function for row-binding with fill
#'
#' @param x First data frame or matrix
#' @param y Second data frame or matrix
#' @param fill Value to use for missing columns
#' @return A combined data frame or matrix
rBind.fill.internal<-function(x,y,fill)
{

  fillMissing<-is.null(fill)
  fill<-if(is(x,'Matrix')) 0 else NA
  if (is.null(nrow(x)))
    x<-matrix(x,nrow=1,dimnames=list(NULL,names(x)))
  if (is.null(nrow(y)))
    y<-matrix(y,nrow=1,dimnames=list(NULL,names(y)))
  y<-
  {
    if('data.frame' %in% is(x) && ('Matrix' %in% is(y)))
      as.data.frame(y)
    else
      if('Matrix' %in% is(x))
        as(y,'Matrix')
    else
      y
  }
  
  nullNames<-FALSE
  #Cannot currently handle duplicate column names
  if(is.null(colnames(x)))
    colnames(x)<-colnames(y)[1:ncol(x)]
  if(is.null(colnames(y)))
    colnames(y)<-colnames(x)[1:ncol(y)]
  if(is.null(colnames(x)))
  {
    nullNames<-TRUE
    colnames(x)<-1:ncol(x)
    colnames(y)<-1:ncol(y)
  }
  ymiss<-colnames(x)[which(is.na(match(colnames(x),colnames(y))))]
  ybind<-rsparsematrix(nrow=nrow(y),ncol=length(ymiss),0)
  colnames(ybind)<-ymiss
  if(!is(y,'Matrix'))
    ybind<-as.matrix(ybind)
  if(!fillMissing)
    ybind[seq_along(ybind)]<-fill
  xmiss<-colnames(y)[which(is.na(match(colnames(y),colnames(x))))]
  xbind<-rsparsematrix(nrow=nrow(x),ncol=length(xmiss),0)
  colnames(xbind)<-xmiss
  if(!is(x,'Matrix')) 
    xbind<-as.matrix(xbind)
  if(!fillMissing)
    xbind[seq_along(xbind)]<-fill
  x<-cbind2(x,xbind)
  y<-cbind2(y,ybind)
  result<-rbind2(x,y[,order(match(colnames(y),colnames(x)))])
  if(nullNames)
    colnames(result)<-NULL
  return(result)
}

#' Load data for the project
#'
#' @param project The project object
#' @param reload Boolean indicating whether to reload data
#' @return A list containing the updated project and loaded data
loadData<-function(project, reload=FALSE)
{

    #require(future)
    #plan(multiprocess)

    #Create empty object house all data sources
    data<-list()
    #For each dataset specified during project design:  
    #1.  Run it's loadData script and populate `data` 
    #2. Save the dataset
    project$datasets<-lapply(project$datasets, function (dataset)
      #future(
      {
        index<-length(data)+1 
        if(is.null(dataset$dataFile) | reload)
        {
          data[[index]]<<-dataset$loadData()
          dir.create('datasets')
          dataset$dataFile<-paste0('datasets/',index,'_',project$timestamp,'.gz')
          saveRDS(data[[index]],dataset$dataFile)
        }
        else {
            flog.info(paste("Loading data source",index,"from file",dataset$dataFile,""))
           data[[index]]<<-readRDS(dataset$dataFile)
    
        }
        return(dataset)
      }
      #)
      )
    
    #Save project periodically.  This, for example, would allow reloading data
    #sources from file rather than primary source
    save.project(project)
    return(list(project,data))
    
}

#' Train the machine learning model
#'
#' @param project The project object
#' @param data The input data
#' @return The updated project object with trained models
train<-function(project,data)
{
  
  
  flog.trace("Received data:")
  flog.trace(capture.output(str(data)))
  
  project$dataStr<-lapply(data,function (x) head(x,1) %>% mutate_if(is.factor, as.character))
  
  require(dplyr)
  require(Matrix.utils)
  
  #The meat of training set creation.  For each dataset, pivot it into a sparse matrix
  #according to the instructions contained in the dataset file 
  nttms<-Map(function (x,y) prepareData(project,x,y),data,project$datasets)
  nttms<-lapply(nttms,function (nttm) {nttm@x[is.na(nttm@x)]<- -1; return(nttm)})
  nttms<-lapply(nttms,function (nttm) nttm[,which(colSums(nttm!=0)>.00001*nrow(nttm))])
  
  flog.trace("Pivoted data:")
  flog.trace(capture.output(str(nttms)))
  
  
  flog.debug('Joining datasets...')  
  
  #The rather complicated process of joining all the sparse matrices together based on dataset keys
  lookupList<-Map(function (dataset,data) unique(data[,c(dataset$key,dataset$alternateKey),drop=FALSE]),project$datasets,data)
  lookup<-Reduce(dplyr::inner_join,lookupList)
  
  lookup<-lookup[match(rownames(nttms[[1]]),lookup[,project$datasets[[1]]$key]),]
  
  if(length(data)>1) {
    nttm<-Reduce(function (x,index) { 
      merge.Matrix(x,nttms[[index]],
                   by.x=lookup[match(rownames(x),lookup[,project$datasets[[1]]$key]), project$datasets[[index]]$key],
                   ,by.y=rownames(nttms[[index]]),all.x=FALSE,all.y=FALSE)
    }, 2:length(data),nttms[[1]])
  }
  #End joins

  
  #Filter out any NAs that emerged from the join process
  nttm@x[is.na(nttm@x)]<- -1
  
  
  #nttm<-lapply(project$datasets,function (dataset) )
  #nttm<-Map(function (dataset,nttms) nttm[lookup[c(dataset$key,dataset$alternateKey)],project$datasets,data) 
  
  # lapply(project[['additionalProcessing']],function (x) eval(x,envir = .GlobalEnv))
  # #source(project$finalPrepFile)
  # lapply(project[['trainProcessing']],function (x) eval(x,envir = .GlobalEnv))
  # 
  
  #nttm<-nttm[,which(colSums(nttm)<(nrow(nttm)-50))] #Warning, this will remove numerics
  
  flog.debug('Adding transforms...')  
  
  #Create transforms on true numeric columns
  cols<-unlist(lapply(project$datasets,function (x) x$columns))
  cols<-intersect(colnames(nttm),cols)
  #Below is better, but too slow and needs work
  #cols<-intersect(colnames(nttm),union(colnames(nttm)[colMeans(nttm)>1],cols))
  
  #Generate transformations based on single columns or combinations of columns
  #For example, linearTransformations will calculate the difference of every 
  transforms<-lapply(project$transforms,function(f, ...) f(...),nttm[,cols])
  #Join in transforms
  nttm<-Reduce(function (x,y) cbind(x,as(y,'dgCMatrix')),transforms,nttm)

  #Remove all columns that don't occur a minimum number of times. 
  flog.debug(paste("Removing columns that don't occur at least",.0001*nrow(nttm),"times."))
  keep<-which(colSums(nttm!=0)>.0001*nrow(nttm))
  flog.debug(paste(ncol(nttm)-length(keep),"columns removed."))
  nttm<-nttm[,keep]
  #Remove columns that are basically all ones
  flog.debug(paste("Inverse of above rule."))
  keep<-which(colSums(nttm==1)<.9999*nrow(nttm))
  flog.debug(paste(ncol(nttm)-length(keep),"columns removed."))
  nttm<-nttm[,keep]
  
  
  #Save final training dataset so that we don't have to do all this again
  project$finalTrainDataFile<-paste0('datasets/nttm',project$timestamp,'.gz')
  
  saveRDS(nttm,file = project$finalTrainDataFile,compress=TRUE)
  
  
  if(is.null(nttm))
    stop("The final data set (nttm) is not populated")
  
  #Order the dataset by the orderBy column (usually a date field).  If this isn't here,
  #logic should exist to order the dataset randomly
  nttm<-nttm[order(nttm[,project$orderBy]),]
  
  #Pull in our outcome/target/response and clean it up a bit.
  project$trainDataOutcome<-nttm[,(project$outcome)]
  
  project$trainDataOutcome<-pmin(project$outcomeMax,pmax(project$outcomeMin,project$trainDataOutcome))
  
  #Remove the outcome column from the training set.
  nttm<-nttm[, colnames(nttm) != project$outcome]
  
  #Break the dataset into train, val, and test.  This area needs a little work.
  project$indices[[1]]<-which(getTestIndex2(nrow(nttm),.1,.6))
  project$indices[[2]]<-which(getTestIndex2(nrow(nttm),.6,.7))
  project$indices[[3]]<-which(getTestIndex2(nrow(nttm),.7,.8))
  #project$indices[[3]]<-which(as.POSIXct(nttm[,'prf_of_dlvr_dt'],origin=lubridate::origin) > (Sys.time() %m+% months(-2)))
  #project$indices[[2]]<-setdiff(which(as.POSIXct(nttm[,'prf_of_dlvr_dt'],origin=lubridate::origin) > (Sys.time() %m+% months(-3))),project$indices[[3]])
  #project$indices[[1]]<-which(as.POSIXct(nttm[,'prf_of_dlvr_dt'],origin=lubridate::origin) <= (Sys.time() %m+% months(-3)))
  
  #Remove elements that occur in both training an validation or testing sets (i.e. same keys, orders, etc)
  #project$indices[[1]]<-project$indices[[1]][!(nttm[project$indices[[1]],project$exclusion] %in% nttm[c(project$indices[[2]],project$indices[[3]]),project$exclusion])]
  flog.debug("Generating folds:")
  #QA check of indices
  flog.debug(capture.output(summary(nttm[project$indices[[1]],project$orderBy])))
  flog.debug(capture.output(summary(nttm[project$indices[[2]],project$orderBy])))
  flog.debug(capture.output(summary(nttm[project$indices[[3]],project$orderBy])))
  
  flog.debug(capture.output(summary(project$trainDataOutcome[project$indices[[1]]])))
  flog.debug(capture.output(summary(project$trainDataOutcome[project$indices[[2]]])))
  flog.debug(capture.output(summary(project$trainDataOutcome[project$indices[[3]]])))
  
  #Remove columns that are unpopulated in either train or val
  flog.debug(paste("Removing columns that don't occur at least",.0001*length(project$indices[[1]]),"times in train."))
  keep<-which(colSums(nttm[project$indices[[1]],]!=0)>.0001*length(project$indices[[1]]))
  flog.debug(paste(ncol(nttm)-length(keep),"columns removed."))
  nttm<- nttm[,keep]
  dim(nttm)
  flog.debug(paste("Removing columns that don't occur at least",.0001*length(project$indices[[2]]),"times in val."))
  keep<-which(colSums(nttm[project$indices[[2]],]!=0)>.0001*length(project$indices[[2]]))
  flog.debug(paste(ncol(nttm)-length(keep),"columns removed."))
  nttm<- nttm[,keep]
  
  dim(nttm)
  
  flog.debug("Removing poison columns...")
  
  a<-abs(colMeans(nttm[project$indices[[1]],])-colMeans(nttm[project$indices[[2]],]))/colSD(nttm)
  flog.debug(paste('Removing columns',paste(colnames(nttm)[which(a>.5)],collapse = ', ')))
  nttm<- nttm[,which(a<=.5)]
  dim(nttm)
  
  project$nttmStr<-nttm[1,,drop=FALSE]
  
  flog.info("Final nttm dimensions:")
  flog.info(dim(nttm))
  
  
#Train models####
  project$models<-lapply(project$models,
                         function (model)
                         {
                           
                           data<-model$preprocess(nttm)
                           model<-model$trainFunc(model,data)
                           model$rawPredictions<-model$predictFunc(model,data)
                           names(model$rawPredictions)<-rownames(nttm)
                           model$metricResults<-lapply(project$metrics,function (f) (f)(model$rawPredictions,model$dataOutcome,data))
                           names(model$metricResults)<-project$metrics
                           model$type<-class(model$model)
                           model$name<-format(Sys.time(),"%Y%m%d%H%M%S")
                           dir.create('savedModels')
                           model$save(model$model,paste0('savedModels/',model$type,model$name))
                           model$modelBinary<-paste0('savedModels/',model$type,model$name)
                           #model$model<-NULL #save space, delete saved model
                           return(model)
                         }
  )
  names(project$models)<-project$modelFiles
  
  #Ensemble####
  project$finalEnsembleModel<-(function (model)
  {
    allPredictions<-data.frame(lapply(project$models,function (x) x$rawPredictions))
    model<-model$trainFunc(model,allPredictions)
    model$name<-format(Sys.time(),"%Y%m%d%H%M%S")
    if(!is.null(model$model))
    {
      model$type<-class(model$model)
      model$save(model$model,paste0('savedModels/',model$type,model$name))
      model$modelBinary<-paste0('savedModels/',model$type,model$name)
    }
    model$predictions<-model$predictFunct(model,allPredictions)
    names(model$predictions)<-rownames(allPredictions)
    return(model)
    
  }) (project$finalEnsembleModel)
  
  project$finalPredictions<-project$finalEnsembleModel$predictions
  
  
  #project$finalEnsembleModel$model<-NULL
  
  
  project$metricsResults<-lapply(project$metrics,function (f) (f)(project$finalPredictions,project$trainDataOutcome,nttm))
  names(project$metricsResults)<-project$metrics  
  
  save.project(project)
  
  flog.info("Train complete.")
  return(project)
    
}

#' Create NTTM (Normalized Term-Term Matrix)
#'
#' @param project The project object
#' @param data The input data
#' @param reload Boolean indicating whether to reload data
#' @return A list containing the updated project and NTTM
createNTTM<-function(project,data,reload=TRUE)
{
  flog.trace("Received data:")
  flog.trace(capture.output(str(data)))
  
  project$dataStr<-lapply(data,function (x) head(x,1) %>% mutate_if(is.factor, as.character))
  
  require(dplyr)
  require(Matrix.utils)
  
  #The meat of training set creation.  For each dataset, pivot it into a sparse matrix
  #according to the instructions contained in the dataset file 
  nttms<-Map(function (x,y) prepareData(project,x,y),data,project$datasets)
  nttms<-lapply(nttms,function (nttm) {nttm@x[is.na(nttm@x)]<- -1; return(nttm)})
  nttms<-lapply(nttms,function (nttm) nttm[,which(colSums(nttm!=0)>.00001*nrow(nttm))])
  
  flog.trace("Pivoted data:")
  flog.trace(capture.output(str(nttms)))
  
  flog.debug('Joining datasets...')  
  
  #The rather complicated process of joining all the sparse matrices together based on dataset keys
  lookupList<-Map(function (dataset,data) unique(data[,c(dataset$key,dataset$alternateKey),drop=FALSE]),project$datasets,data)
  lookup<-Reduce(dplyr::inner_join,lookupList)
  
  lookup<-lookup[match(rownames(nttms[[1]]),lookup[,project$datasets[[1]]$key]),]
  
  if(length(data)>1) {
    nttm<-Reduce(function (x,index) { 
      merge.Matrix(x,nttms[[index]],
                   by.x=lookup[match(rownames(x),lookup[,project$datasets[[1]]$key]), project$datasets[[index]]$key],
                   ,by.y=rownames(nttms[[index]]),all.x=FALSE,all.y=FALSE)
    }, 2:length(data),nttms[[1]])
  } else
    nttm<-nttms[[1]]
  
  #End joins
  
  
  #Filter out any NAs that emerged from the join process
  nttm@x[is.na(nttm@x)]<- -1
  
  
  #nttm<-nttm[,which(colSums(nttm)<(nrow(nttm)-50))] #Warning, this will remove numerics
  
  flog.debug('Adding transforms...')  
  
  #Create transforms on true numeric columns
  cols<-unlist(lapply(project$datasets,function (x) x$columns))
  cols<-intersect(colnames(nttm),cols)
  #Below is better, but too slow and needs work
  #cols<-intersect(colnames(nttm),union(colnames(nttm)[colMeans(nttm)>1],cols))
  
  #Generate transformations based on single columns or combinations of columns
  #For example, linearTransformations will calculate the difference of every 
  transforms<-lapply(project$transforms,function(f, ...) f(...),nttm[,cols])
  #Join in transforms
  nttm<-Reduce(function (x,y) cbind(x,as(y,'dgCMatrix')),transforms,nttm)

  #Order the dataset by the orderBy column (usually a date field).  If this isn't here,
  #logic should exist to order the dataset randomly
  nttm<-nttm[order(nttm[,project$orderBy]),]
  
  return(list(project,nttm))
}

#' Train the machine learning model
#'
#' @param project The project object
#' @param nttm The Normalized Term-Term Matrix
#' @return The updated project object with trained models
train<-function(project,nttm)
{
  if(is.null(nttm))
    stop("The final data set (nttm) is not populated")
  
  require(Matrix.utils)
  require(dplyr)
  #Break the dataset into train, val, and test.  This area needs a little work.
  project$indices[[1]]<-which(getTestIndex2(nrow(nttm),.1,.6))
  project$indices[[2]]<-which(getTestIndex2(nrow(nttm),.6,.8))
  project$indices[[3]]<-which(getTestIndex2(nrow(nttm),.8,.9))

  # 
  # project$indices<-tapply(1:nrow(nttm),nttm[,'index'],list)
  
  #Pull in our outcome/target/response and clean it up a bit.
  project$trainDataOutcome<-nttm[,(project$outcome)]
  
  project$trainDataOutcome<-pmin(project$outcomeMax,pmax(project$outcomeMin,project$trainDataOutcome))
  
  #Remove the outcome column from the training set.
  nttm<-nttm[, colnames(nttm) != project$outcome]
  
 
  #Remove elements that occur in both training an validation or testing sets (i.e. same keys, orders, etc)
  #project$indices[[1]]<-project$indices[[1]][!(nttm[project$indices[[1]],project$exclusion] %in% nttm[c(project$indices[[2]],project$indices[[3]]),project$exclusion])]
  flog.debug("Generating folds:")
  #QA check of indices
  flog.debug(capture.output(summary(nttm[project$indices[[1]],project$orderBy])))
  flog.debug(capture.output(summary(nttm[project$indices[[2]],project$orderBy])))
  flog.debug(capture.output(summary(nttm[project$indices[[3]],project$orderBy])))

  flog.debug(capture.output(summary(project$trainDataOutcome[project$indices[[1]]])))
  flog.debug(capture.output(summary(project$trainDataOutcome[project$indices[[2]]])))
  flog.debug(capture.output(summary(project$trainDataOutcome[project$indices[[3]]])))

  #Remove columns that are unpopulated in either train or val
  flog.debug(paste("Removing columns that don't occur at least",.0001*length(project$indices[[1]]),"times in train."))
  keep<-which(colSums(nttm[project$indices[[1]],]!=0)>.0001*length(project$indices[[1]]))
  flog.debug(paste(ncol(nttm)-length(keep),"columns removed."))
  nttm<- nttm[,keep]
  dim(nttm)
  flog.debug(paste("Removing columns that don't occur at least",.0001*length(project$indices[[2]]),"times in val."))
  keep<-which(colSums(nttm[project$indices[[2]],]!=0)>.0001*length(project$indices[[2]]))
  flog.debug(paste(ncol(nttm)-length(keep),"columns removed."))
  nttm<- nttm[,keep]
  
  dim(nttm)
  
  flog.debug("Removing poison columns...")

  #These are heuristics that exist to filter out columns that have properties harmful to training
  #Heuristic 1
  goodCor<-t(nttm) %*% project$trainDataOutcome
  goodCor<-goodCor[,1]
  goodCor<-goodCor/(sqrt(colSums(nttm^2)*sum(project$trainDataOutcome^2)))
  
  badCor<-t(nttm) %*% nttm[,project$orderBy]
  badCor<-badCor[,1]
  badCor<-badCor/(sqrt(colSums(nttm^2)*sum(nttm[,project$orderBy]^2)))
  
  #Heuristic 2
  flog.debug(paste('Removing columns',paste(colnames(nttm)[which(abs(badCor) > abs(goodCor) & abs(badCor) > .6)],collapse = ', ')))  
  nttm<-nttm[,!(abs(badCor) > abs(goodCor) & abs(badCor) > .6)]  
  
  #Heuristic 3
  a<-abs(colMeans(nttm[project$indices[[1]],])-colMeans(nttm[project$indices[[2]],]))/colSD(nttm)
  flog.debug(paste('Removing columns',paste(colnames(nttm)[which(a>.5)],collapse = ', ')))
  nttm<- nttm[,which(a<=.5)]
  dim(nttm)

    
  project$nttmStr<-nttm[1,,drop=FALSE]
  
  flog.info("Final nttm dimensions:")
  flog.info(dim(nttm))

  
  #Train models####
  project$models<-lapply(project$models,
                         function (model)
                         {

                           data<-model$preprocess(nttm)
                           model<-model$trainFunc(model,data)
                           model$rawPredictions<-model$predictFunc(model,data)
                           names(model$rawPredictions)<-rownames(nttm)
                           model$metricResults<-lapply(project$metrics,function (f) (f)(model$rawPredictions,model$dataOutcome,data))
                           names(model$metricResults)<-project$metrics
                           model$type<-class(model$model)
                           model$name<-format(Sys.time(),"%Y%m%d%H%M%S")
                           dir.create('savedModels')
                           model$save(model$model,paste0('savedModels/',model$type,model$name))
                           model$modelBinary<-paste0('savedModels/',model$type,model$name)
                           #model$model<-NULL #save space, delete saved model
                           return(model)
                         }
                         )
  names(project$models)<-project$modelFiles

  #Ensemble####
  project$finalEnsembleModel<-(function (model)
  {
    allPredictions<-data.frame(lapply(project$models,function (x) x$rawPredictions))
    model<-model$trainFunc(model,allPredictions)
    model$name<-format(Sys.time(),"%Y%m%d%H%M%S")
    if(!is.null(model$model))
    {
      model$type<-class(model$model)
      model$save(model$model,paste0('savedModels/',model$type,model$name))
      model$modelBinary<-paste0('savedModels/',model$type,model$name)
    }
    model$predictions<-model$predictFunct(model,allPredictions)
    names(model$predictions)<-rownames(allPredictions)
    return(model)
    
  }) (project$finalEnsembleModel)
  
  project$finalPredictions<-project$finalEnsembleModel$predictions
  
  
  #project$finalEnsembleModel$model<-NULL
  
  
  project$metricsResults<-lapply(project$metrics,function (f) (f)(project$finalPredictions,project$trainDataOutcome,nttm))
  names(project$metricsResults)<-project$metrics  
  
  flog.info("Train complete.")
  return(project)
  
}

#' Create a new project
#'
#' @return A new project object
create<-function()
{
  
  wd<-paste0(choose.dir(default = getwd(),caption='Set working project directory.'),'\\')
  setwd(wd)
  #Project setup####
  project<-NULL
  project$timestamp<-format(Sys.time(),"%Y%m%d%H%M%S")
  project$name<-paste0('project',project$timestamp)
  
  project$setupFile<-browse.files(multi = FALSE)
  #Remove static path information
  project$setupFile<-gsub(wd,'',project$setupFile,fixed=TRUE)
  #Switch DOS format file names to Linux
  project$setupFile<-unlist(gsub('\\\\','/',project$setupFile))
  
  source(project$setupFile,echo = TRUE,verbose = TRUE)
  
#Load Data####
  project$loadDataFiles<-NULL
  while({print("Add any data load scripts? Y/N");readline()}=='Y')
    project$loadDataFiles<-c(project$loadDataFiles,browse.files(caption='Choose data load scripts'))
  
  #Remove static path information
  project$loadDataFiles<-lapply(project$loadDataFiles,function (x) gsub(wd,'',x,fixed=TRUE))
  project$loadDataFiles<-unlist(lapply(project$loadDataFiles,function (x) gsub('\\\\','/',x)))
  
  #Run this to reload load data source from files
  project$datasets<-lapply(project$loadDataFiles,
                         function (loadDataFile)
                         {
                           dataset<-NULL
                           source(loadDataFile,local = TRUE)  
                           return(dataset)
                         }
  )
  names(project$datasets)<-project$loadDataFiles
  
  save.project(project)
  # 
  # #Preprocess####
  # project$preprocessFiles<-NULL
  # while({print("Add new preprocess files? Y/N");readline()}=='Y')
  #   project$preprocessFiles<-c(project$preprocessFiles,browse.files(caption='Choose preprocess files',default='config/preprocess/'))
  # 
  # project$preprocess<-unlist(lapply(project$preprocessFiles, function (file)
  # {
  #   source(file,local=TRUE)
  #   fnames <- lsf.str(-1)
  #   return(mget(fnames))
  # }
  # ))
  
  #Transforms####
  project$transformFiles<-NULL
  while({print("Add new transform files? Y/N");readline()}=='Y')
    project$transformFiles<-c(project$transformFiles,browse.files(caption='Choose transform files',default='config/transforms/'))
  
  project$transformFiles<-lapply(project$transformFiles,function (x) gsub(wd,'',x,fixed=TRUE))
  project$transformFiles<-unlist(lapply(project$transformFiles,function (x) gsub('\\\\','/',x)))
  
  project$transforms<-unlist(lapply(project$transformFiles, function (file)
  {
    source(file,local=TRUE)
    fnames <- lsf.str(-1)
    return(mget(fnames))
  }
  ))
  
  # #Additional Processing - to be run after data[[1]] preprocessing####
  # project$additionalProcessingFiles<-NULL
  # while({print("Add any additional files? Y/N");readline()}=='Y')
  #   project$additionalProcessingFiles<-c(project$additionalProcessingFiles,browse.files(caption='Choose additional files'))
  # 
  # project$additionalProcessing<-lapply(project$additionalProcessingFiles,parse)
  # 
  # #Train processing - only run in training phase####
  # project$trainProcessingFiles<-NULL
  # while({print("Add any final train processing files? Y/N");readline()}=='Y')
  #   project$trainProcessingFiles<-c(project$trainProcessingFiles,browse.files(caption='Choose additional files'))
  # 
  # project$trainProcessing<-lapply(project$trainProcessingFiles,parse)
  
  #Metrics####
  project$metricsFiles<-NULL
  while({print("Add new metrics files? Y/N");readline()}=='Y')
    project$metricsFiles<-c(project$metricsFiles,browse.files(caption='Choose metrics files'))
  
  project$metricsFiles<-lapply(project$metricsFiles,function (x) gsub(wd,'',x,fixed=TRUE))
  project$metricsFiles<-unlist(lapply(project$metricsFiles,function (x) gsub('\\\\','/',x)))
  
  
  project$metrics<-unlist(lapply(project$metricsFiles, function (file)
  {
    source(file,local=TRUE)
    fnames <- lsf.str(-1)
    return(mget(fnames))
  }
  ))
  
  #Models####
  project$modelFiles<-NULL
  while({print("Add any model files? Y/N");readline()}=='Y')
    project$modelFiles<-c(project$modelFiles,browse.files(caption='Choose model files'))
  
  project$modelFiles<-lapply(project$modelFiles,function (x) gsub(wd,'',x,fixed=TRUE))
  project$modelFiles<-unlist(lapply(project$modelFiles,function (x) gsub('\\\\','/',x)))
  
  #Run this to reload model scripts from files
  project$models<-NULL
  project$models<-lapply(project$modelFiles,
                         function (modelFile)
                         {
                           modelTmp<-NULL
                           source(modelFile,local = TRUE,)  
                           return(modelTmp)
                         }
  )
  names(project$models)<-project$modelFiles
  
  #Ensemble - to be run after one or more models have been added to `models`####
  project$finalEnsembleFile<-browse.files(multi = FALSE)
  
  
  project$finalEnsembleFile<-gsub(wd,'',project$finalEnsembleFile,fixed=TRUE)
  project$finalEnsembleFile<-unlist(gsub('\\\\','/',project$finalEnsembleFile))
  
  project$finalEnsembleModel<-(function (modelFile)
  {
    model<-NULL
    source(modelFile,local = TRUE)    
    return(model)
  }) (project$finalEnsembleFile)
  
  # #Postprocess - to be run after everything else####
  # project$postprocessFiles<-NULL
  # while({print("Add new postprocess files? Y/N");readline()}=='Y')
  #   project$postprocessFiles<-c(project$postprocessFiles,browse.files(caption='Choose postprocess files',default='config/postprocess/'))
  # 
  # project$postprocess<-unlist(lapply(project$postprocessFiles, function (file)
  # {
  #   source(file,local=TRUE)
  #   fnames <- lsf.str(-1)
  #   return(mget(fnames))
  # }
  # ))
  
  save.project(project)
  return(project)
  
}

#' Score the model
#'
#' @param project The project object
#' @param data The input data
#' @return Predictions
score<-function(project,data)
{
  require(futile.logger)
  flog.trace('Received data:')
  flog.trace(capture.output(str(data)))
  require(Matrix.utils)
  
  env<-environment()
  
  nttm<-NULL
  
  flog.debug('Preparing data...')  
  
  nttms<-Map(function (x,y) prepareData(project,x,y),data,project$datasets)
# 
  flog.trace("Pivoted data:")
  flog.trace(capture.output(str(nttms)))
  
  flog.debug('Joining datasets...')  
  lookupList<-Map(function (dataset,data) data[,c(dataset$key,dataset$alternateKey),drop=FALSE],project$datasets,data)
  lookup<-Reduce(dplyr::left_join,lookupList)
  
  lookup<-lookup[match(rownames(nttms[[1]]),lookup[,project$datasets[[1]]$key]),]
  
  if(length(data)>1)
    nttm<-Reduce(function (x,index) { 
      merge.Matrix(x,nttms[[index]],
                   by.x=lookup[match(rownames(x),lookup[,project$datasets[[1]]$key]), project$datasets[[index]]$key],
                   ,by.y=rownames(nttms[[index]]),all.x=FALSE,all.y=FALSE)
    }, 2:length(data),nttms[[1]])
  
#   flog.debug('Additional processing...')  
#   lapply(project$additionalProcessing,function (x) eval(as.expression(unlist(x)),envir = env))

  nttm@x[is.na(nttm@x)]<-0
  
  if(is.null(nttm))
    stop("The final data set (nttm) is not populated")
 
  flog.debug('Adding transforms...')  
  cols<-unlist(lapply(project$datasets,function (x) x$columns))
  cols<-intersect(colnames(nttm),cols)
  transforms<-lapply(project$transforms,function(f, ...) f(...),nttm[,cols])
  #Join in transforms
  nttm<-Reduce(function (x,y) cbind(x,as(y,'dgCMatrix')),transforms,nttm) 

  #remove extra columns
  nttm<-nttm[,intersect(colnames(nttm),colnames(project$nttmStr)),drop=FALSE]
    
  #Add in missing columns
  nttm<-rBind.fill(project$nttmStr,nttm)[-1,,drop=FALSE]
  
  nttm[is.na(nttm)]<-0
  
  
  # if(is.null(models))
  #   models<-lapply(project$models,function (x) load(x$modelBinary,envir = env))
  # flog.trace('Final dataset:')
  
  flog.debug('Making predictions...')
  results<-data.frame(lapply(project$models,function (model) model$predictFunc(model,nttm)))
  flog.debug('Ensembling...')
  predictions<-project$finalEnsembleModel$predictFunct(project$finalEnsembleModel,results)
  flog.info("Final predictions complete:")
  flog.info(capture.output(summary(predictions)))
  #predictions<-Reduce(function(f, ...) f(...),rev(project$postprocess),predictions,data2,nttm,right=TRUE)
  #Manual Reduce to overcome argument shortcoming in builtin Reduce function
  
  if(!isTRUE(nrow(nttm)==length(predictions)))
    flog.error('Nttm and predictions are not the same length.')
  
  for (i in seq_along(project$postprocess)) 
    predictions <- forceAndCall(2, function(f, ...) f(...), project$postprocess[[i]], predictions, data2, nttm)
  
  names(predictions)<-rownames(nttm)
  
  return(predictions)
  
  
}

#' Package the project
#'
#' @param project The project object
#' @param score.only Boolean indicating whether to package only scoring components
#' @param include.libraries Boolean indicating whether to include libraries
#' @return None
package.project<-function(project,score.only=TRUE,include.libraries=TRUE)
{
  message('Creating project directory...')
  name<-format(Sys.time(),"%Y%m%d%H%M%S")
  dir.create(name)
  
  message('Copying models...')
  dir.create(paste(name,'savedModels',sep='\\'))
  modelBinaries<-unlist(lapply(project$models,function (x) x$modelBinary))
  file.copy(modelBinaries,paste(name,'savedModels',sep='\\'))
  file.copy(project$finalEnsembleModel$modelBinary,paste(name,'savedModels',sep='\\'))
  
  message('Copying project...')
  file.copy(paste0('projects/',project$name),name)
  file.copy(paste0('projects/',project$name,'.desc'),name)
  
  setwd(name)
  message('Copying libraries...')
  #libraryDir<-paste(name,'libraries',sep='\\')
  dir.create('libraries')
  snapshotPackages('libraries')
  #dump(c('test','prepareData'),paste0(name,'/functions.R'))
  dump(c('test','prepareData','load.project'),'functions.R')
  
  
  zip1Name<-paste0(project$name,'_',name,'.zip')
  zip(paste('..',zip1Name,sep='\\'),'*')
  
  #zip(paste('..',zip1Name,sep='\\'),c(zip1Name,'..\\deployRInit.R','..\\deployRTest.R'),flags='-j')

  setwd('..')
    
  unlink(name,recursive = TRUE,force=TRUE)
  
}

#' Snapshot packages
#'
#' @param to Directory to save package snapshots
#' @return None
snapshotPackages<-function(to)
{
  require(miniCRAN)
  #This could be made much faster using zip, unzip, but would be uglier
  libraries<-rev(.libPaths())
  loadedPackages<-pkgDep((.packages()),suggests=FALSE)
  lapply(libraries,function (directory)
    {
      lapply(loadedPackages,function (package)
        {
          message(paste('Copying', package,',,,'))
          file.copy(paste(directory,package,sep='/'),to = to,recursive = TRUE, overwrite = TRUE)
        }  
      )  
    }
  )
}

#' Prepare data for modeling
#'
#' @param project The project object
#' @param dat The input data
#' @param dataset The dataset configuration
#' @return Prepared data
prepareData<-function(project,dat,dataset)
{
  # if(!is.null(project$preprocess))
  #   data2<-Reduce(function(f, ...) f(...),rev(project$preprocess),data[[1]], right=TRUE)
  # 
  # #Profile data
  # #data2 %>% str
  # #tableplot2(data2,nCols = 20)
  # 
  # transforms<-lapply(project$transforms,function(f, ...) f(...),data2[,project$predictorColumns])
  # if(isTRUE(nrow(transforms)==nrow(data2)))
  #   data2<-data.frame(data2,transforms)

  require(Matrix.utils)
require(Matrix)
  
  #If the dataset is not already a sparse matrix, turn it in to one.  The data will be pivoted out based on key.
  
  if (!is(dat,"Matrix"))
  {
    pivot<-NULL
    pivots<-NULL
    #This is for regular pivots that preserve numerics and one-hot encode factors
    if(!is.null(dataset$columns))
    {
      form<-as.formula(paste(dataset$key," ~ ", paste(dataset$columns, collapse= "+")))
      pivot<-dMcast(dat,form,fun.aggregate = 'mean')
      #pivot@x[is.na(pivot@x)]<- -1
      #pivot<-pivot[,which(colSums(pivot)>.0001*nrow(pivot))]
    }
    #This is for pivots that instead of one-hot encoding factors aggregate some other numeric.  
    #Ex.  Dollars spent by LOB, ship qty by sku, etc.
    if(!is.null(dataset$numerics))
    {
      pivots<-lapply(dataset$numerics, function (col) 
        {
          dat[is.na(dat[,col]),col]<-0
          form2<-as.formula(paste(dataset$key," ~ ", paste(dataset$factors, collapse= "+")))
          pivot2<-dMcast(dat,form2,value.var = col,fun.aggregate = 'mean')
          #pivot2@x[is.na(pivot2@x)]<- -1
#pivot2<-pivot2[,which(colSums(pivot2)>.0001*nrow(pivot2))]
          colnames(pivot2)<-paste0(colnames(pivot2),col)
          return(pivot2)
        }
      )
    }
    result<-do.call(cbind,c(pivot,pivots) %>% as.list())
  }
  return(result)
  
}

#' Load a project from file
#'
#' @param fileName Name of the file to load the project from
#' @return The loaded project object
load.project<-function(fileName)
{
  #Should global environment be used here to make sure that everything (models, datasets, project, etc) are loaded into memory?
  #Alternatives?  Pass the entire environment back?
  project<-readRDS(fileName)
  
  if(length(project$models) > 0)
    if(is.character(project$models[[1]]$modelBinary))
      project$models<-lapply(project$models,function (model) {model$model<-model$load(model$modelBinary); return(model)})
  if(is.character(project$finalEnsembleModel$modelBinary))
    project$finalEnsembleModel$model<-project$finalEnsembleModel$load(project$finalEnsembleModel$modelBinary)
  return(project)
}

#' Browse files
#'
#' @param ... Additional arguments passed to file browsing function
#' @return Selected file path(s)
browse.files<-function(...)
{
  #if(interactive())
  #  if(!isTRUE(require(rChoiceDialogs))) {
  #    result<-choose.files(...) }
  #  else 
  #    result<-rchoose.files(...)
  #else
    result<-textFileBrowser(...)
    
    return(result)
    
}

#' Text-based file browser
#'
#' @param root Root directory to start browsing from
#' @param multiple Boolean indicating whether multiple file selection is allowed
#' @param ... Additional arguments
#' @return Selected file path(s)
textFileBrowser<- function (root=getwd(), multiple=F,...) {
  # .. and list.files(root)
  x <- c( dirname(normalizePath(root)), list.files(root,full.names=T) )
  isdir <- file.info(x)$isdir
  obj <- sort(isdir,index.return=T,decreasing=T)
  isdir <- obj$x
  x <- x[obj$ix]
  lbls <- sprintf('%s%s',basename(x),ifelse(isdir,'/',''))
  lbls[1] <- sprintf('../ (%s)', basename(x[1]))
  
  files <- c()
  sel = -1
  while ( TRUE ) {
    sel <- menu(lbls,title=sprintf('Select file(s) (0 to quit) in folder %s:',root))
    if (sel == 0 )
      break
    if (isdir[sel]) {
      # directory, browse further
      files <- c(files, textFileBrowser( x[sel], multiple ))
      break
    } else {
      # file, add to list
      files <- c(files,x[sel])
      if ( !multiple )
        break
      # remove selected file from choices
      lbls <- lbls[-sel]
      x <- x[-sel]
      isdir <- isdir[-sel]
    }
  }
  return(files)
}

#' Save project
#'
#' @param project The project object to save
#' @param overwriteName Boolean indicating whether to overwrite existing project file
#' @return None
save.project<-function(project, overwriteName=is.null(project$datasets[[1]]$dataFile) && is.null(project$models[[1]]$modelBinary))
{

  project<-as.list(project)
  name<-project$name
  project$lastUpdated<-format(Sys.time(),"%Y%m%d%H%M%S")
  if(!overwriteName)
    name<-paste(name,project$lastUpdated,sep='_')
  # if(!overwriteName & !is.null(project$finalPredictions))
  #   name<-paste(name,project$lastTrain,sep='_')
  dir.create('projects')
  project$models<-lapply(project$models, function (model) {model$model<-NULL; return(model)}) #save space, delete saved model
  project$finalEnsembleModel$model<-NULL
  sink(file = paste0('projects/',name,'.desc'),append = FALSE)
  str(project)
  sink(NULL)
  saveRDS(project,file = paste0('projects/',name),compress=TRUE)
  
}

#' Get dataset from database
#'
#' @param table Table name
#' @param query SQL query
#' @param dsn Data source name
#' @param type Connection type ('JDBC', 'RSQLServer', or 'odbc')
#' @param drv Database driver
#' @param uri Database URI
#' @return Retrieved dataset
getDataSet<-function(table='',query=paste('select * from',table),dsn='btccmart',type='odbc',drv=NULL,uri=NULL)
{
  require(futile.logger)
  flog.info(paste('Retrieving: ',query))
  if(type=='JDBC')
  {
    xj<-dbConnect(drv,uri)
    data<-dbGetQuery(xj,query)
    dbDisconnect(xj)
  } else
    if(type=='RSQLServer')
    {
      require(RSQLServer)
      xj<-dbConnect(RSQLServer::SQLServer(), dsn)
      cursor<-dbSendQuery(xj,query)
      data<-dbFetch(cursor)
      
    } else
      if(type=='odbc')
      {
        require(odbc)
        
        xj<-dbConnect(odbc::odbc(),dsn)
        data<-dbGetQuery(xj,query)
        dbDisconnect(xj)
      }
  else
  {
    require(RODBC)
    xj<-odbcConnect(dsn)
    data<-sqlQuery(xj,query,as.is=TRUE)
  }
  message(head(data))
  #data<-data.frame(lapply(data,function (x) if (is.character(x)) as.factor(x) else x))
  data<-convertToDates(data)
  data[,sapply(data,is.character)]<-lapply(data[,sapply(data,is.character),drop=FALSE],type.convert)
  data<-data.frame(lapply(data,function (x) {if (is.factor(x)) levels(x)<-stringr::str_trim(levels(x)); return (x)}))
  data.table::setnames(data,tolower(colnames(data)))
  flog.info(paste(nrow(data),'rows retrieved.'))
  
  return(data)
}

#' Convert columns to dates
#'
#' @param data Input data frame
#' @return Data frame with date columns converted
convertToDates<-function(data)
{
  require(lubridate)
  dates<-lapply(data,function (x) try(as.POSIXct(x[sample.int(length(x),pmin(10000,length(x)))]),silent=TRUE))
  dateCols<-which(unlist(lapply(dates,function (x) inherits(x,'POSIXt'))))
  replace<-data.frame(lapply(data[,dateCols,drop=FALSE],function (x) lubridate::parse_date_time(x,orders=c('YmdHMOS','Ymd'),tz = 'US/Central')))
  data[,dateCols]<-replace
  return(data)
}

#' Convert character columns to factors
#'
#' @param data Input data frame
#' @return Data frame with character columns converted to factors
convertToFactors<-function(data)
{
  charCols<-unlist(lapply(data,is.character))
  replace<-lapply(data[,charCols],factor)
  data[,charCols]<-replace
  return(data)
}

#' Create table plot
#'
#' @param data Input data frame
#' @param nCols Number of columns to display in each plot
#' @param ... Additional arguments passed to tableplot
#' @return None
tableplot2<-function(data,nCols=ncol(data),...)
{
  require(tabplot)
  breaks<-ceiling(ncol(data)/nCols)
  index<-2:ncol(data)
  cuts<-cut(index,breaks,labels=FALSE)
  lapply(unique(cuts),function (colSet) tableplot(data[,c(1,index[cuts==colSet])],sample=TRUE,plot=TRUE,decreasing = FALSE))
}

#' Train a boosted model
#'
#' @param train Training data
#' @param dataOutcome Outcome variable
#' @param trainIndex Training set indices
#' @param valIndex Validation set indices
#' @param params Model parameters
#' @return Trained model
trainBoostModel<-function(train,dataOutcome,trainIndex,valIndex,params=NULL)
{
  require(xgboost)

  
  dtrain <- xgb.DMatrix(train[trainIndex,,drop=FALSE], label=dataOutcome[trainIndex])
  
  dval <- xgb.DMatrix(train[valIndex,,drop=FALSE], label=dataOutcome[valIndex])
  
  watchlist <- list(train=dtrain,eval = dval)
  
  #The meat.  Train gradient boosted trees on training set. 

  model<-xgb.train(data= dtrain
                   
                   ,nrounds             = 300 #300, #280, #125, #250, # changed from 300
                   ,early_stopping_rounds =  10
                   ,watchlist           = watchlist
                   #,maximize            = FALSE
                   ,print_every_n = 5
                   ,params=params     
  )
  

  return(model)
  
}

#' Get test index
#'
#' @param range Range of indices
#' @param first Number of indices to select from the beginning
#' @param last Number of indices to select from the end
#' @param sample Number of indices to sample from the middle
#' @return Test indices
getTestIndex<-function(range,first,last,sample)
{
  trainIndex<-range
  testIndex<-head(trainIndex,first)
  trainIndex<-setdiff(trainIndex,testIndex)
  testIndex<-c(testIndex,tail(trainIndex,last))
  trainIndex<-setdiff(trainIndex,testIndex)
  testIndex<-c(testIndex,sample(trainIndex,sample))
  return(testIndex)
}

#' Get test index (alternative method)
#'
#' @param n Total number of indices
#' @param start Start proportion (default 0)
#' @param end End proportion (default length(range))
#' @return Logical vector of test indices
getTestIndex2<-function(n,start=0,end=length(range))#,first,last,sample)
{
  
  index<-logical(n)
  if(start < 1& start > 0)
    start<-start*n+1
  if(end < 1& end > 0)
    end<-end*n
  index[start:end]<-TRUE
  return(index)
}

#' Replace all NAs in a data frame
#'
#' @param data Input data frame
#' @return Data frame with NAs replaced
replaceAllNAS<-function(data)
{
  data<-data.frame(lapply(data,replaceNAS))
  return(data)
}

#' Replace NAs in a column
#'
#' @param column Input column
#' @param replacement Replacement value for NAs (default "Other")
#' @return Column with NAs replaced
replaceNAS<-function(column,replacement="Other")
{
  
  if(inherits(column,'POSIXt'))
    return(column)
  if(inherits(column,'Date'))
    return(column)
  #column2<-as.character(column) #was too slow
  if(is.factor(column))
  {
    if(any(is.na(column) | is.infinite(column) | is.nan(column)))
      attr(column,'levels')<-union(levels(column),replacement)
  }
  #Adding highly controversial decision to cast infinite and NaN as 0
  #Slightly slow operation
  column[is.na(column) | is.infinite(column) | is.nan(column)]<-ifelse(is.numeric(column),0,replacement)
  return(column)
}

#' Remove empty columns from a data frame
#'
#' @param data Input data frame
#' @param num Minimum number of unique values required to keep a column
#' @return Data frame with empty columns removed
removeEmptyColumns<-function(data,num=2)
{
  #Slow operation
  #Should not be used on test data
  result<-data
  #result<-data[,unlist(lapply(data,function (x) length(unique(x))) >= num)]
  return(result)
}

#' Optimize hyperparameters
#'
#' @param train Training data
#' @param dataOutcome Outcome variable
#' @param trainIndex Training set indices
#' @param valIndex Validation set indices
#' @param fitness Fitness function (optional)
#' @return Optimized hyperparameters
optimizeHyperParameters<-function(train,dataOutcome,trainIndex,valIndex,fitness=NULL)
{
  library(GA)
  
  # if(is.null(fitness))
  #   fitness<-function(...)
  #   {
  #     params=as.list(...)
  #     names(params)<-c('eta','max_depth','colsample_bytree','min_child_weight','gamma','alpha','lambda','lambda_bias')
  #     params$max_depth<-round(params$max_depth)
  #     model<-xgb.train(data= dtrain
  #                      ,subsample           = min(.5,10000/nrow(train))
  #                      ,nrounds             = 500 #300, #280, #125, #250, # changed from 300
  #                      ,watchlist           = watchlist
  #                      ,early_stopping_round = 6
  #                      ,params = params
  #     )
  #     return(0-model$best_score)
  #   }
  # 
  
  fitness<-function(...)
  {
    require(dplyr)
    params=as.list(...)
    names(params)<-c('eta','max_depth','colsample_bytree','min_child_weight','gamma','alpha')
    params$max_depth<-round(params$max_depth)
    model<-xgb.train(data= dtrain
                     ,subsample           = min(.5,10000/nrow(train))
                     ,nrounds             = 500 #300, #280, #125, #250, # changed from 300
                     ,watchlist           = watchlist
                     ,early_stopping_round = 6
                     ,eval_metric='mae'
                     ,params = params
    )
    predictions<-predict(model,dval)
    performance<-table(floor(predictions)==getinfo(dval,'label')) %>% prop.table %>% .['TRUE']
    print('Exact on-time rate:')
    print(performance)
    if(isTRUE(performance>.22))
      browser()
    return(performance)
  }
  
require(xgboost)
  
  dtrain <- xgb.DMatrix(train[trainIndex,,drop=FALSE], label=dataOutcome[trainIndex])
  
  dval <- xgb.DMatrix(train[valIndex,,drop=FALSE], label=dataOutcome[valIndex])
  
  watchlist <- list(train = dtrain,eval = dval)
  
  
  #    GA<-ga(type = "real-valued", fitness = fitness,
  #           min = c(0.00001, 1, .001, 1, 0, 0, 0, 0), max = c(1, 100, 1, 5000, 1, 50, 50, 50)
  #           ,popSize = 15, crossover = gareal_blxCrossover, maxiter = 20,run = 3, names = c('eta','max_depth','colsample_bytree','min_child_weight','gamma','alpha','lambda','lambda_bias'))
  #
  
  GA<-ga(type = "real-valued", fitness = function (...) fitness(...),
         min = c(0.00001, 1, .01, 1, 0, 0), max = c(1, 100, 1, 5000, 1, 50),popSize = pmax(60,pmin(1000000000/(as.numeric(ncol(train))*nrow(train)),100))
         , crossover = gareal_blxCrossover
         , selection = gareal_tourSelection
         , mutation = gareal_rsMutation
         #, pmutation=.2
         ,maxiter = 9,run = 3, names = c('eta','max_depth','colsample_bytree','min_child_weight','gamma','alpha'))
  browser()
  save(GA,file='GA.data')
  
  GAModels1<-apply(GA@population[GA@fitness>=sort(GA@fitness,decreasing=TRUE)[15],]
                   ,1
                   ,function (pop) 
                   { 
                     pop<-as.list(pop) ; 
                     names(pop)<-c('eta','max_depth','colsample_bytree','min_child_weight','gamma','alpha') ;
                     pop$max_depth<-round(pop$max_depth)
                     model<-xgb.train(data= dtrain
                                      
                                      ,subsample           = min(.5,50000/nrow(train))
                                      ,nrounds             = 1500 #300, #280, #125, #250, # changed from 300
                                      ,watchlist           = watchlist
                                      ,maximize            = TRUE
                                      ,early.stop.round = 40
                                      ,params = pop
                     )
                     
                   }
  )
  
  
  scores<-lapply(GAModels1,function (x) x$bestScore) %>% unlist
  scores>=.99*max(scores)
  
  GAModels1<-GAModels1[scores>=.99*max(scores)]
  
  GAModels2<-apply(GA@population[GA@fitness>=sort(GA@fitness,decreasing=TRUE)[15],]
                   ,1
                   ,function (pop) 
                   { 
                     pop<-as.list(pop) ; 
                     names(pop)<-c('eta','max_depth','colsample_bytree','min_child_weight','gamma','alpha') ;
                     pop$max_depth<-round(pop$max_depth)
                     model<-xgb.train(data= dtrain
                                      ,objective           = "binary:logistic"
                                      ,subsample           = 0.6
                                      ,eval_metric         = "auc"
                                      #,feature_names=as.character(colnames(val))
                                      ,nrounds             = 1500 #300, #280, #125, #250, # changed from 300
                                      ,watchlist           = watchlist
                                      ,maximize            = TRUE
                                      ,early.stop.round = 40
                                      #,verbose=0
                                      #,silent=1
                                      ,params = pop
                     )
                     
                   }
  )
  scores<-lapply(GAModels2,function (x) x$bestScore) %>% unlist
  scores>=.99*max(scores)
  
  GAModels2<-GAModels2[scores>=.99*max(scores)]
  
}

#' Create a Shiny dashboard
#'
#' @return None
createShiny<-function()
{
  require(shiny)
  require(shinydashboard)
  
  ui <- dashboardPage(
    header=dashboardHeader(title = "Basic dashboard"),
    sidebar=dashboardSidebar(),
    body=dashboardBody(# Boxes need to be put in a row (or column)
      fluidRow
      (
        box(plotOutput("plot1"),height=300),
        box(plotOutput("plot3"),height=300)
      ),
      fluidRow(
        box(plotOutput("plot2",width=1400))
      )
    )
  )
  
  server <- function(input, output) {
    
    #This is needed because multiple lines per order number skews the results
    tmpData<-data2[data2$ord_nbr %in% sample(unique(data2$ord_nbr),.1*nrow(data2)),] %>%
      group_by(ord_nbr) %>%
      summarize(ord_dt=max(ord_dt),week_dlvr=max(week_dlvr),dlvr_day_cnt=max(dlvr_day_cnt),estd_dlvr_day_cnt=max(estd_dlvr_day_cnt))
    
    output$plot1 <- renderPlot({
      ggplot()+
        stat_smooth(data=tmpData,aes(x=ord_dt,y=dlvr_day_cnt),geom='point',color='red')+
        stat_smooth(data=sample_frac(qa2,size=.1),aes(x=ord_dt,y=dlvr_dt_bus_day_cnt-ord_dt_bus_day_cnt),color='blue',geom='point')+
        scale_x_datetime(date_breaks = '1 month',limits = c(max(data2$ord_dt) %m+% months(-15),max(data2$ord_dt)))+
        theme_bw()+
        theme(axis.text.x=element_text(angle = 45,hjust=1))
      
      
    },height=300)
    
    
    output$plot2<-renderPlot({tableplot(sample_frac(data2[,c('ordinal_dt',predictorColumns)],.1),decreasing = FALSE)})
    
    lateTest<-data.frame(tmpData[,c("week_dlvr","dlvr_day_cnt","estd_dlvr_day_cnt")],late=tmpData$estd_dlvr_day_cnt<tmpData$dlvr_day_cnt)
    lateSum<-lateTest %>% group_by(week_dlvr) %>% summarize(lates=sum(late),total=n())
    
    plot3<- ggplot() + 
      geom_line(data=lateSum,aes(x=week_dlvr,y=lates/total),color='red')+
      geom_line(data=official,aes(x=Week.of.Deliver.Date,y=Late),color='blue')+
      scale_x_datetime(date_breaks = '1 month',limits = c(max(data2$ord_dt) %m+% months(-15),max(data2$ord_dt)))+
      theme_bw()+
      theme(axis.text.x=element_text(angle = 45,hjust=1))
    
    if(isTRUE(try(!is.null(project$finalPredictions))))
    {
      preds<-data.frame(data2[,c('ord_nbr','week_dlvr','dlvr_day_cnt')],prediction=project$finalPredictions) %>%
        .[data2$ord_nbr %in% sample(unique(data2$ord_nbr),.1*nrow(data2)),] %>%
        group_by(ord_nbr) %>%
        summarize(week_dlvr=max(week_dlvr),prediction=max(prediction),dlvr_day_cnt=max(dlvr_day_cnt))
      preds$late<-preds$prediction<preds$dlvr_day_cnt
      preds<-preds %>%
        group_by(week_dlvr) %>%
        summarize(lates=sum(late),total=n())
      plot3<-plot3 + geom_line(data=preds,aes(x=week_dlvr,y=lates/total),color='dark green')
    }
    
    output$plot3<-renderPlot({plot3},height=300) 
  }
  
  app<-shinyApp(ui, server)
  runApp(app,launch.browser =TRUE)
}

#' Calculate column standard deviation
#'
#' @param x Input matrix
#' @param na.rm Boolean indicating whether to remove NA values
#' @return Vector of column standard deviations
colSD<- function(x, na.rm=TRUE) {
  # if (na.rm) {
  #   n <- colSums(!is.na(x)) # thanks @flodel
  # } else {
  #   n <- nrow(x)
  # }
  n<-nrow(x)
  colVar <- colMeans(x^2, na.rm=na.rm) - (colMeans(x, na.rm=na.rm))^2
  return(sqrt(colVar * n/(n-1)))
}

#' Transform and reduce data
#'
#' @return None
transformReduce<-function()
{
  cols<-unlist(lapply(project$datasets,function (x) x$columns))

  a<-nttm-nttm[,'estd_bus_dlvr_dt']
  system.time(b<-(a^2)/sqrt(nttm[,intersect(colnames(nttm),cols)]^2*nttm[,'estd_bus_dlvr_dt']^2))
b[!is.finite(b)]<-0
b[1:2,(colMeans(b) < 1)]

  
}

#' Filter bad columns
#'
#' @param dat Input data
#' @param trainIndex Training set indices
#' @param valIndex Validation set indices
#' @return Filtered data
filterBadColumns<-function(dat, trainIndex, valIndex)
{
  flags<-numeric(nrow(dat))
  flags[trainIndex]<- -.5
  flags[valIndex]<- .5
    dat<-cbind(dat,flags)

  dat<-dat[union(trainIndex,valIndex),]
  sdBased<-abs(colMeans(dat[dat[,'flags']==-.5,])-colMeans(dat[dat[,'flags']==.5,]))/colSD(dat)
  cors1<-colSums(dat*dat[,'flags'])/colSums(sqrt(dat^2*dat[,'flags']^2))
  cors2<-colSums(abs(dat*dat[,'flags']))/colSums(sqrt(dat^2*dat[,'flags']^2))
  cors3<-sweep(as.matrix(dat),2,dat[,'flags'],cor)
  system.time(cors4<-apply(dat[sample(1:nrow(dat),10000),],2,function (x) grr::sort2(x)))

}

#' Test data condition
#'
#' @param dat Input data
#' @param condition Condition to test
#' @param threshold Threshold function
#' @param message Message to display
#' @param connection Database connection (optional)
#' @param tableName Table name for writing results (optional)
#' @param filter Boolean indicating whether to filter data based on test
#' @return Filtered data if filter is TRUE, otherwise input data
test_it<-function(dat,condition,threshold,message,connection=NULL,tableName=NULL,filter=FALSE)
{
  require(futile.logger)
  tests<-eval(substitute(condition), dat)
  results<-capture.output(summary(tests))
  if(threshold(tests)) {
    flog.error(message)
    flog.error(results)
  } else {
    flog.debug(message)
    flog.debug(results)
  }
  if(filter) {
    if(!is.null(connection) && mean(tests) <1) {
      #schema<-DBI::dbGetQuery(connection,paste('select top 1 * from',tablename)[0,]
      fails<-dat[!tests,]
      fails$project<-project$name
      fails$message<-message
      fails$timestamp<-Sys.time()
      #DBI::dbWriteTable(connection,tableName,fails,append=TRUE)
      append<-DBI::dbExistsTable(connection,tableName)
      DBI::dbWriteTable(connection,tableName,fails,append=append)
    }
    dat<-dat[tests,]
  }
  return(dat)
}

#' Target encode a column
#'
#' @param col Input column
#' @param target Target variable
#' @return Encoded column
targetEncode<-function(col,target)
{
  #sums<-tapply(target,col,sum)
  #counts<-tapply(target,col,length)
  result<-reorder(col,target,mean)
  #means<-tapply(target,col,mean)
  
}

# Set coercion methods for various data types
setAs('character','POSIXct',function (from) as.POSIXct(from))
setAs('character','factor',function (from) as.factor(from))
setAs('logical','factor',function (from) as.factor(from))
setAs('logical','POSIXct',function (from) as.POSIXct(from))
setAs('Date','POSIXct',function (from) as.POSIXct(from))