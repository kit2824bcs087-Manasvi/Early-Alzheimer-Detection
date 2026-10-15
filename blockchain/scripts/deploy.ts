import { network } from "hardhat";

const connection = await network.connect();

const ethers = connection.ethers;

console.log("Deploying AlzoraRegistry...");

const AlzoraRegistry = await ethers.getContractFactory(
  "AlzoraRegistry"
);

const registry = await AlzoraRegistry.deploy();

await registry.waitForDeployment();

const address = await registry.getAddress();

console.log("AlzoraRegistry deployed successfully!");
console.log("Contract Address:", address);