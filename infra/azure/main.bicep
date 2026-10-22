@description('Azure region')
param location string = resourceGroup().location

@description('Globally unique suffix')
param suffix string

@description('Container image for the FastAPI service')
param apiImage string

@description('Azure OpenAI endpoint for recommendation generation')
param azureOpenAIEndpoint string = ''

@description('Azure OpenAI API key for recommendation generation')
@secure()
param azureOpenAIApiKey string = ''

@description('Azure OpenAI deployment name for recommendation generation')
param azureOpenAIDeployment string = ''

@description('Azure OpenAI API version for recommendation generation')
param azureOpenAIApiVersion string = '2024-06-01'

@description('Azure OpenAI endpoint for retrieval embeddings')
param retrievalAzureOpenAIEndpoint string = ''

@description('Azure OpenAI API key for retrieval embeddings')
@secure()
param retrievalAzureOpenAIApiKey string = ''

@description('Azure OpenAI deployment name for retrieval embeddings')
param retrievalAzureOpenAIDeployment string = 'text-embedding-3-large'

@description('Azure OpenAI API version for retrieval embeddings')
param retrievalAzureOpenAIApiVersion string = '2024-02-15-preview'

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'retention-logs-${suffix}'
  location: location
  properties: {
    retentionInDays: 30
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'retention-env-${suffix}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: listKeys(logs.id, logs.apiVersion).primarySharedKey
      }
    }
  }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'retain${suffix}'
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'retention-api-${suffix}'
  location: location
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          env: [
            {
              name: 'APP_ENV'
              value: 'azure'
            }
            {
              name: 'AGENT_PROVIDER'
              value: 'azure_openai'
            }
            {
              name: 'ALLOW_DETERMINISTIC_FALLBACK'
              value: 'true'
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAIEndpoint
            }
            {
              name: 'AZURE_OPENAI_API_KEY'
              value: azureOpenAIApiKey
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT'
              value: azureOpenAIDeployment
            }
            {
              name: 'AZURE_OPENAI_API_VERSION'
              value: azureOpenAIApiVersion
            }
            {
              name: 'RETRIEVAL_EMBEDDING_PROVIDER'
              value: 'azure_openai'
            }
            {
              name: 'RETRIEVAL_AZURE_OPENAI_ENDPOINT'
              value: retrievalAzureOpenAIEndpoint
            }
            {
              name: 'RETRIEVAL_AZURE_OPENAI_API_KEY'
              value: retrievalAzureOpenAIApiKey
            }
            {
              name: 'RETRIEVAL_AZURE_OPENAI_DEPLOYMENT'
              value: retrievalAzureOpenAIDeployment
            }
            {
              name: 'RETRIEVAL_AZURE_OPENAI_API_VERSION'
              value: retrievalAzureOpenAIApiVersion
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 3
      }
    }
  }
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output storageAccountName string = storage.name
